from __future__ import annotations

import asyncio
import logging

from ..bus import CHOICES, Assessment, Bus
from ..llm import DecisionModel
from ..state import NewLap, StateStore

log = logging.getLogger(__name__)


class Analyst:
    """Background agent. Two stages: calculate parameters, then judge them.

    Subclasses implement `measure()` (cheap maths over the state) and may
    implement `on_lap()`. The decision model (or a threshold fallback) turns
    the numbers plus `instructions` into ignore/mention/urgent.
    """

    name = "analyst"
    topic = "general"
    instructions = ""
    interval = 1.0  # seconds between measure() calls

    def __init__(self, store: StateStore, bus: Bus, decision: DecisionModel):
        self.store, self.bus, self.decision = store, bus, decision

    def measure(self) -> tuple[dict, str] | None:
        """Return (context, message) or None when there is nothing to assess."""
        return None

    def measure_lap(self, lap: NewLap) -> tuple[dict, str] | None:
        return None

    async def judge_and_publish(self, result: tuple[dict, str] | None) -> None:
        if result is None:
            return
        context, message = result
        level = await self.decision.decide(self.instructions, context, CHOICES)
        t = self.store.car.t if self.store.car else 0.0
        self.bus.publish(Assessment(self.name, self.topic, level, message, context, t))

    async def run(self) -> None:
        while True:
            try:
                await self.judge_and_publish(self.measure())
            except Exception:
                log.exception("%s failed", self.name)
            await asyncio.sleep(self.interval)

    async def handle_lap(self, lap: NewLap) -> None:
        try:
            await self.judge_and_publish(self.measure_lap(lap))
        except Exception:
            log.exception("%s lap handler failed", self.name)
