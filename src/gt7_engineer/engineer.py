"""The race engineer: the only voice. Arbitrates analyst assessments and talks with the driver."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Awaitable, Callable

from . import ui
from .briefing import lap_update
from .bus import PRIORITY, URGENT, Assessment, Bus
from .llm import MAX_REPLY_TOKENS, ChatClient
from .state import NewLap, StateStore

log = logging.getLogger(__name__)

SYSTEM = (
    "You are the driver's race engineer, on the radio during a Gran Turismo 7 race. The driver is busy "
    "driving, so be concise and accurate: one short sentence, two at most. Lead with the answer or the "
    "action, then the key number. No greetings, filler, pleasantries, apologies or explanations unless asked. "
    "Plain spoken words, no markdown, no emoji. Say lap times the way a person would speak them, rounded "
    "to the tenth of a second, e.g. 'one thirty-two point four'.\n"
    "The driver speaks to you and a speech-to-text system transcribes it, so expect mistakes: wrong or "
    "similar-sounding words ('feel' for 'fuel', 'tires' for 'tyres', 'lap time' heard as 'lap tame'), missing "
    "words, odd punctuation. You know it is a race, so read between the lines: work out what the driver most "
    "likely meant and answer that, without pointing out the mistake. Only ask them to repeat if it truly "
    "could mean several different things.\n"
    "Use the numbers in the data and never invent any. You may and should do simple arithmetic on them "
    "(differences, totals, per-lap averages, laps remaining times burn) to answer what was asked; the data "
    "includes fuel at race start, fuel burned so far, fuel at the start of each lap, and the lap times. "
    "Only say you don't have something when it genuinely cannot be worked out from the data "
    "(for example gaps to other cars).\n"
    "Examples of the style (the numbers are made up; always use the real data):\n"
    "Driver: how's my fuel? -> 'Four point two laps of fuel, five to go. Save fuel or box.'\n"
    "Driver: last lap? -> 'One forty-one point nine, half a second off your best.'\n"
    "Driver: how much feel have I used? -> 'Thirty-one litres so far, seven per lap.'\n"
    "Driver: gap to the leader? -> 'Don't have that.'"
)

TYRE_WORDS = re.compile(r"\b(tyres?|tires?|wear|fronts?|rears?)\b", re.I)
TYRE_QUESTION = "How are the tyres looking?"
COVERED_BY_UPDATE = {"fuel", "pace"}  # the lap update already reports these
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

    def reset(self) -> None:
        """A new race started: drop everything that belonged to the last one."""
        self.pending.clear()
        self.history.clear()
        self.awaiting_tyre_reply = False
        self._urgent_at.clear()
        self._last_spoke = float("-inf")

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
                    {"role": "user", "content": f"Radio the driver this in one short sentence, keeping every number: {facts} {extra}".strip()},
                ],
                max_tokens=MAX_REPLY_TOKENS,
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
        """Every new lap gets a race update: built from the race state, plus anything else the analysts have."""
        await asyncio.sleep(1.0)  # let the analysts publish for this lap
        items = sorted(self.pending.values(), key=lambda a: -PRIORITY[a.level])
        self.pending.clear()
        # Fuel and pace are already in the update itself; only add what it doesn't cover (e.g. coaching).
        extras = [a.message for a in items if a.topic not in COVERED_BY_UPDATE][:2]
        parts = [lap_update(self.store, event), *extras]
        if self.tyre_ask_every and event.record.lap % self.tyre_ask_every == 0:
            parts.append(TYRE_QUESTION)
            self.awaiting_tyre_reply = True
        await self._say(" ".join(p for p in parts if p))

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
                max_tokens=MAX_REPLY_TOKENS,
            )
            return text or "Sorry, say again?"
        except Exception as e:
            log.warning("engineer answer failed: %s", e)
            return "Sorry, I lost the model there. Say again in a moment."
