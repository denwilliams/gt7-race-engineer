"""Wires telemetry -> state -> analysts -> race engineer -> voice."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable

from . import ui
from .ui import fmt_lap as ui_fmt
from .agents.base import Analyst
from .agents.fuel import FuelAgent, fuel_rule
from .agents.lap_trace import LapTraceAgent
from .agents.pace import PaceAgent, pace_rule
from .bus import Bus
from .config import Config
from .engineer import RaceEngineer
from .llm import ChatClient, JevDecision, OpenAIDecision, ThresholdDecision, make_chat
from .state import NewRace, StateStore, TrackMapped
from .telemetry.packet import Frame
from .telemetry.stats import IngestStats

log = logging.getLogger(__name__)


class Interrupter:
    """Lets the microphone (created first) reach the engineer (created inside run())."""

    def __init__(self) -> None:
        self.handler: Callable[[], None] | None = None

    def fire(self) -> None:
        if self.handler:
            self.handler()


async def run(
    cfg: Config,
    source: AsyncIterator[Frame],
    speak: Callable[[str], Awaitable[None]],
    utterances: AsyncIterator[str] | None = None,
    use_llm: bool = True,
    speaking: asyncio.Event | None = None,
    stop_speaking: Callable[[], None] | None = None,
    interrupter: Interrupter | None = None,
    stats: IngestStats | None = None,
    reset_request: asyncio.Event | None = None,
    recorder=None,
) -> StateStore:
    store, bus = StateStore(mirror=cfg.track.mirror), Bus()
    stats = stats if stats is not None else IngestStats()
    ui.attach(store=store, stats=stats)

    async def speak_muted(text: str) -> None:
        if speaking is not None:
            speaking.set()
        try:
            await speak(text)
        finally:
            if speaking is not None:
                speaking.clear()

    engineer_client = make_chat(cfg.engineer) if use_llm else None
    reviewer_client = make_chat(cfg.lap_reviewer) if use_llm else None
    jev = JevDecision(cfg.decision, "ignore") if use_llm and cfg.decision and cfg.decision.kind == "jev" else None
    decision_client = ChatClient(cfg.decision) if use_llm and cfg.decision and not jev else None

    def decision_for(rule, default: str):
        if jev:
            return jev
        return OpenAIDecision(decision_client, default) if decision_client else ThresholdDecision(rule)

    analysts: list[Analyst] = [
        FuelAgent(store, bus, decision_for(fuel_rule, "ignore")),
        PaceAgent(store, bus, decision_for(pace_rule, "ignore")),
    ]
    lap_trace = LapTraceAgent(store, bus, reviewer_client) if reviewer_client else None
    engineer = RaceEngineer(
        store, bus, engineer_client, speak_muted, min_gap=cfg.min_seconds_between_calls, stop_speaking=stop_speaking
    )
    if interrupter is not None:
        interrupter.handler = engineer.interrupt

    tasks = [asyncio.create_task(a.run()) for a in analysts]
    tasks.append(asyncio.create_task(engineer.run()))
    background: set[asyncio.Task] = set()

    def spawn(coro) -> None:
        t = asyncio.create_task(coro)
        background.add(t)

        def done(task: asyncio.Task) -> None:
            background.discard(task)
            if not task.cancelled() and task.exception() is not None:  # never fail silently
                log.error("background task failed: %r", task.exception(), exc_info=task.exception())

        t.add_done_callback(done)

    heard: asyncio.Queue[str] = asyncio.Queue()

    async def listen() -> None:
        # Only capture here. Handling a sentence (LLM + speaking, 10 s+) must not hold up the
        # microphone loop, or audio backs up and is processed late, echo and all.
        try:
            async for text in utterances:
                heard.put_nowait(text)
        except Exception:
            log.exception("microphone/speech recognition stopped; the engineer can no longer hear you")

    async def respond() -> None:
        while True:
            await engineer.on_driver(await heard.get())

    if utterances is not None:
        tasks.append(asyncio.create_task(listen()))
        tasks.append(asyncio.create_task(respond()))

    last_lap_seen = None
    try:
        async for frame in source:
            stats.frame(frame.packet_id)
            if recorder is not None:
                recorder.write(frame)
            if frame.lap != last_lap_seen:  # what the game says, before any of our logic touches it
                ui.log(
                    "system",
                    f"lap counter {last_lap_seen} -> {frame.lap} (running: {frame.on_track}, flags 0x{frame.flags & 0xFFFF:04x}, "
                    f"last lap {frame.last_lap_ms} ms)",
                )
                last_lap_seen = frame.lap
            if reset_request is not None and reset_request.is_set():  # the driver asked (keyboard)
                reset_request.clear()
                store.reset()
                engineer.reset()
                ui.log("system", "🔄 Race state reset (you pressed r)")
            event = store.update(frame)
            if isinstance(event, NewRace):
                engineer.reset()
                ui.log("system", f"🔄 New race detected ({event.reason}): state reset")
                event = None
            while store.events:  # pit stops and track-map changes since the last frame
                stop = store.events.popleft()
                if isinstance(stop, TrackMapped):
                    ui.log("system", f"🗺 Track mapped: {stop.turns} turns (from {stop.laps} clean lap{'s' if stop.laps != 1 else ''})")
                    continue
                bits = ([f"+{stop.added_l:.0f} L"] if stop.added_l else []) + (["new tyres"] if stop.tyres_changed else [])
                ui.log("lap", f"🛞 Pit stop, lap {stop.lap}: {', '.join(bits)}")
                spawn(engineer.on_pit_stop(stop))
            if event:
                rec = event.record
                ui.log("lap", f"Lap {rec.lap}: {ui_fmt(rec.time_ms)}, fuel used {rec.fuel_used:.1f} L")
                for a in analysts:
                    spawn(a.handle_lap(event))
                if lap_trace:
                    spawn(lap_trace.handle_lap(event))
                spawn(engineer.on_lap(event))
        await asyncio.sleep(2.0)  # let final lap handlers finish
    finally:
        for t in tasks:
            t.cancel()
        for c in (engineer_client, reviewer_client, decision_client, jev):
            if c:
                await c.aclose()
    return store
