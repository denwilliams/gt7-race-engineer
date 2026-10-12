"""The race engineer: the only voice. Arbitrates analyst assessments and talks with the driver."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Awaitable, Callable

from . import ui
from .bus import PRIORITY, URGENT, Assessment, Bus
from .llm import ChatClient
from .state import NewLap, StateStore

log = logging.getLogger(__name__)

SYSTEM = (
    "You are the driver's race engineer in Gran Turismo 7, on the radio. Be brief: one or two short spoken "
    "sentences, plain words, no markdown, no emoji. Use only the facts given; never invent numbers. "
    "If you don't know, say so."
)

TYRE_WORDS = re.compile(r"\b(tyres?|tires?|wear|fronts?|rears?)\b", re.I)
TYRE_QUESTION = "How are the tyres looking?"
ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
    "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15,
    "sixteenth": 16,
}
_ORD = "|".join(ORDINALS)


def parse_position(text: str) -> int | None:
    """Race position the driver states: 'P3', 'position 3', 'I'm in third', '3rd place'. None for questions."""
    if "?" in text or QUESTION.match(text):
        return None
    t = text.lower()
    m = (
        re.search(r"\bp\s?(\d{1,2})\b", t)
        or re.search(r"\b(?:position|place)\s+(?:number\s+)?(\d{1,2})\b", t)
        or re.search(r"\b(\d{1,2})(?:st|nd|rd|th)\b", t)
    )
    if m:
        return int(m.group(1))
    m = re.search(rf"\b({_ORD})\b(?:\s+place|\s+position)?", t)
    if m and re.search(r"\b(i'?m|i am|in|currently|now|running|sitting|position|place)\b", t):
        return ORDINALS[m.group(1)]
    return None


QUESTION = re.compile(r"\s*(how|what|when|where|why|which|who|can|could|should|do|does|is|are|will)\b", re.I)


class RaceEngineer:
    def __init__(
        self,
        store: StateStore,
        bus: Bus,
        client: ChatClient | None,
        speak: Callable[[str], Awaitable[None]],
        min_gap: float = 8.0,
        tyre_ask_every: int = 3,
        stop_speaking: Callable[[], None] | None = None,
    ):
        self.store, self.bus, self.client, self.speak = store, bus, client, speak
        self.stop_speaking = stop_speaking
        self._saying = False
        self._interrupted = False
        self.min_gap = min_gap
        self.tyre_ask_every = tyre_ask_every
        self.pending: dict[str, Assessment] = {}  # latest per topic
        self.history: list[dict] = []
        self.awaiting_tyre_reply = False
        self._last_spoke = float("-inf")
        self._urgent_at: dict[str, float] = {}  # topic -> game time of last urgent call
        self.urgent_cooldown = 30.0
        self._speaking = asyncio.Lock()

    # ---- helpers ------------------------------------------------------
    def _now(self) -> float:
        return self.store.car.t if self.store.car else 0.0

    def interrupt(self) -> None:
        """The driver started talking over the engineer: cut the speech off now."""
        if self._saying and self.stop_speaking is not None:
            self._interrupted = True
            self.stop_speaking()

    async def _say(self, text: str) -> None:
        async with self._speaking:
            log.debug("ENGINEER: %s", text)
            self._last_spoke = self._now()
            self.history.append({"role": "assistant", "content": text})
            self._saying, self._interrupted = True, False
            ui.log("engineer", text)
            ui.set_engineer("speaking")
            try:
                await self.speak(text)
            finally:
                self._saying = False
                ui.set_engineer("idle")
                if self._interrupted:  # the driver only heard part of it
                    self.history[-1] = {"role": "assistant", "content": f"{text} [cut off by the driver]"}

    async def _phrase(self, items: list[Assessment], extra: str = "") -> str:
        facts = " ".join(a.message for a in items)
        if self.client is None:
            return f"{facts} {extra}".strip()
        ui.set_engineer("thinking")
        try:
            text = await self.client.chat(
                [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"Radio the driver these facts: {facts} {extra}".strip()},
                ],
                max_tokens=80,
            )
            return text or facts
        except Exception as e:
            log.warning("engineer LLM failed, using raw facts: %s", e)
            return f"{facts} {extra}".strip()

    # ---- assessments --------------------------------------------------
    async def run(self) -> None:
        """Drain analyst assessments; speak urgent ones now, queue the rest for the lap update."""
        while True:
            a = await self.bus.queue.get()
            ui.log("analyst", f"[{a.level}] {a.message}")
            if a.topic in self.pending and PRIORITY[self.pending[a.topic].level] > PRIORITY[a.level]:
                continue  # keep the more severe one
            self.pending[a.topic] = a
            if (
                a.level == URGENT
                and self._now() - self._last_spoke >= 2.0
                and self._now() - self._urgent_at.get(a.topic, float("-inf")) >= self.urgent_cooldown
            ):
                self._urgent_at[a.topic] = self._now()
                self.pending.pop(a.topic, None)
                await self._say(await self._phrase([a]))

    async def on_lap(self, event: NewLap) -> None:
        """Lap ping: give one combined update, then maybe ask about tyres."""
        await asyncio.sleep(1.0)  # let the analysts publish for this lap
        items = sorted(self.pending.values(), key=lambda a: -PRIORITY[a.level])
        self.pending.clear()
        ask = self.tyre_ask_every and event.record.lap % self.tyre_ask_every == 0
        if not items and not ask:
            return
        if items and self._now() - self._last_spoke < self.min_gap and not ask:
            self.pending.update({a.topic: a for a in items})  # too soon; keep for next lap
            return
        text = await self._phrase(items[:3])
        if ask:
            text = f"{text} {TYRE_QUESTION}".strip()
            self.awaiting_tyre_reply = True
        await self._say(text)

    # ---- driver -------------------------------------------------------
    async def on_driver(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        log.debug("DRIVER: %s", text)
        self.history.append({"role": "user", "content": text})
        pos = parse_position(text)
        if pos is not None and not self.awaiting_tyre_reply:
            self.store.add_report("position", f"P{pos} at lap {self.store.lap}")
            await self._say(f"Copy, P{pos}.")
            return
        if self.awaiting_tyre_reply or (TYRE_WORDS.search(text) and not QUESTION.match(text) and "?" not in text):
            self.awaiting_tyre_reply = False
            self.store.add_report("tyre_wear", text)
            await self._say("Copy, tyres noted.")
            return
        await self._say(await self._answer(text))

    async def _answer(self, question: str) -> str:
        state = json.dumps(self.store.summary())
        if self.client is None:
            return "No language model is configured, so I can't answer that."
        ui.set_engineer("thinking")
        try:
            text = await self.client.chat(
                [{"role": "system", "content": f"{SYSTEM}\nCurrent car and race state: {state}"}]
                + self.history[-8:],
                max_tokens=100,
            )
            return text or "Sorry, say again?"
        except Exception as e:
            log.warning("engineer answer failed: %s", e)
            return "Sorry, I lost the model there. Say again in a moment."
