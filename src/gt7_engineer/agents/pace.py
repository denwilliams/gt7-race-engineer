from __future__ import annotations

import statistics

from ..bus import IGNORE, MENTION
from ..state import NewLap
from .base import Analyst


def pace_rule(ctx: dict) -> str:
    if ctx.get("new_best"):
        return MENTION
    return MENTION if ctx["delta_to_best_s"] > 1.0 or ctx["delta_to_typical_s"] > 1.5 else IGNORE


def fmt(ms: int) -> str:
    m, s = divmod(ms / 1000, 60)
    return f"{int(m)}:{s:06.3f}"


class PaceAgent(Analyst):
    name = "pace"
    topic = "pace"
    interval = 3600  # lap-driven only
    instructions = (
        "You assess lap pace. delta_to_best_s is how much slower the last lap was than the session best, "
        "delta_to_typical_s versus your typical lap (the median of recent clean laps). A new personal best is worth mentioning; "
        "a lap more than about a second off pace is worth mentioning; otherwise ignore."
    )

    def measure_lap(self, lap: NewLap):
        t = lap.record.time_ms
        if t <= 0:
            return None
        times = [l.time_ms for l in self.store.laps if l.time_ms > 0 and l is not lap.record]
        best = min(times) if times else None
        # "typical" = the median of recent clean laps: in-laps, out-laps and mistakes don't move it
        clean = [l.time_ms for l in self.store.laps if l.time_ms > 0 and l is not lap.record and l.burn_valid]
        avg = statistics.median(clean[-5:]) if clean else None
        ctx = {
            "lap": lap.record.lap,
            "lap_time": fmt(t),
            "new_best": best is None or t < best,
            "delta_to_best_s": round((t - best) / 1000, 2) if best else 0.0,
            "delta_to_typical_s": round((t - avg) / 1000, 2) if avg else 0.0,
        }
        if best is None:
            msg = f"Lap {ctx['lap']}: {ctx['lap_time']}, first timed lap."
        elif ctx["new_best"]:
            msg = f"Lap {ctx['lap']}: {ctx['lap_time']}, new best by {-ctx['delta_to_best_s']:.2f}s."
        else:
            msg = f"Lap {ctx['lap']}: {ctx['lap_time']}, {ctx['delta_to_best_s']:+.2f}s to best, {ctx['delta_to_typical_s']:+.2f}s to your typical lap."
        return ctx, msg
