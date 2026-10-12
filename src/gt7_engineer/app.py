"""Wires telemetry -> state -> analysts -> race engineer -> voice."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable

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

log = logging.getLogger(__name__)


async def run(
    cfg: Config,
    source: AsyncIterator[Frame],
    speak: Callable[[str], Awaitable[None]],
    utterances: AsyncIterator[str] | None = None,
    use_llm: bool = True,
    speaking: asyncio.Event | None = None,
) -> StateStore:
    store, bus = StateStore(), Bus()

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
    engineer = RaceEngineer(store, bus, engineer_client, speak_muted, min_gap=cfg.min_seconds_between_calls)

    tasks = [asyncio.create_task(a.run()) for a in analysts]
    tasks.append(asyncio.create_task(engineer.run()))
    background: set[asyncio.Task] = set()

    def spawn(coro) -> None:
        t = asyncio.create_task(coro)
        background.add(t)
        t.add_done_callback(background.discard)

    async def listen() -> None:
        async for text in utterances:
            await engineer.on_driver(text)

    if utterances is not None:
        tasks.append(asyncio.create_task(listen()))

    try:
        async for frame in source:
            event = store.update(frame)
            if event:
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
