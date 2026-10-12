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
from .llm import ChatClient, JevDecision, OpenAIDecision, ThresholdDecision
from .state import StateStore
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
) -> StateStore:
    store, bus = StateStore(), Bus()
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

    engineer_client = ChatClient(cfg.engineer) if use_llm else None
    reviewer_client = ChatClient(cfg.lap_reviewer) if use_llm else None
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
        t.add_done_callback(background.discard)

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

    try:
        async for frame in source:
            stats.frame()
            event = store.update(frame)
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
