"""Reviews a completed lap's telemetry with an LLM and publishes coaching."""

from __future__ import annotations

import logging

from ..bus import MENTION, Assessment, Bus
from ..llm import ChatClient
from ..state import LapRecord, NewLap, StateStore
from ..telemetry.packet import Frame

log = logging.getLogger(__name__)

SYSTEM = (
    "You are a racing data analyst reviewing one lap of Gran Turismo 7 telemetry for the race engineer. "
    "You are given a compact trace of the lap and, if available, the driver's best lap for comparison. "
    "Reply with at most two short sentences naming the single most useful thing to improve "
    "(where time was lost: braking too early/late, low minimum speed, late throttle). "
    "Only state what the data shows. If nothing stands out, reply exactly: nothing notable."
)


def trace(frames: list[Frame], buckets: int = 20) -> list[dict]:
    """Downsample a lap to `buckets` evenly spaced samples (by time)."""
    if not frames:
        return []
    step = max(1, len(frames) // buckets)
    out = []
    for i in range(0, len(frames), step):
        w = frames[i : i + step]
        out.append(
            {
                "pct": round(100 * i / len(frames)),
                "kph": round(sum(f.speed_kph for f in w) / len(w)),
                "thr": round(max(f.throttle for f in w), 2),
                "brk": round(max(f.brake for f in w), 2),
                "gear": w[-1].gear,
            }
        )
    return out[:buckets]


def fastest(laps, exclude: LapRecord) -> LapRecord | None:
    timed = [l for l in laps if l is not exclude and l.time_ms > 0]
    return min(timed, key=lambda l: l.time_ms) if timed else None


class LapTraceAgent:
    name = "lap_trace"

    def __init__(self, store: StateStore, bus: Bus, client: ChatClient):
        self.store, self.bus, self.client = store, bus, client

    def prompt(self, lap: LapRecord) -> str:
        parts = [f"Lap {lap.lap}, time {lap.time_ms} ms.\nTrace: {trace(lap.frames)}"]
        ref = fastest(self.store.laps, lap)
        if ref:
            parts.append(f"Best lap ({ref.time_ms} ms) trace: {trace(ref.frames)}")
        return "\n".join(parts)

    async def handle_lap(self, event: NewLap) -> None:
        try:
            text = await self.client.chat(
                [{"role": "system", "content": SYSTEM}, {"role": "user", "content": self.prompt(event.record)}],
                max_tokens=120,
            )
        except Exception as e:
            log.warning("lap review failed: %s", e)
            return
        if "nothing notable" in text.lower():
            return
        t = self.store.car.t if self.store.car else 0.0
        self.bus.publish(Assessment(self.name, "coaching", MENTION, text, {"lap": event.record.lap}, t))
