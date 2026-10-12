"""Assessments published by analyst agents and consumed by the race engineer."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

IGNORE, MENTION, URGENT = "ignore", "mention", "urgent"
CHOICES = [IGNORE, MENTION, URGENT]
PRIORITY = {URGENT: 2, MENTION: 1, IGNORE: 0}


@dataclass(slots=True)
class Assessment:
    source: str  # agent name
    topic: str  # e.g. "fuel"; used to de-duplicate
    level: str  # ignore | mention | urgent
    message: str  # short factual text for the engineer to phrase
    data: dict = field(default_factory=dict)
    t: float = 0.0  # game-clock time


class Bus:
    def __init__(self) -> None:
        self.queue: asyncio.Queue[Assessment] = asyncio.Queue()

    def publish(self, a: Assessment) -> None:
        if a.level != IGNORE:
            self.queue.put_nowait(a)
