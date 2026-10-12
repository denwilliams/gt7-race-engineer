"""Live car/race state with a rolling frame history that can be queried back in time."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .telemetry.packet import Frame

FRAME_HZ = 60
HISTORY_SECONDS = 600


@dataclass(slots=True)
class LapRecord:
    lap: int
    time_ms: int  # as reported by the game; -1 if unknown
    fuel_used: float
    frames: list[Frame] = field(repr=False, default_factory=list)


@dataclass(slots=True)
class DriverReport:
    t: float
    kind: str  # e.g. "tyre_wear"
    value: str  # free text, e.g. "fronts 60%, rears 70%"


@dataclass(slots=True)
class NewLap:
    """Event emitted by StateStore.update() when a lap completes."""

    record: LapRecord


class StateStore:
    def __init__(self, history_seconds: int = HISTORY_SECONDS, keep_laps: int = 10):
        self.frames: deque[Frame] = deque(maxlen=history_seconds * FRAME_HZ)
        self.laps: deque[LapRecord] = deque(maxlen=keep_laps)
        self.reports: list[DriverReport] = []
        self.car: Frame | None = None
        self._lap_frames: list[Frame] = []
        self._lap_start_fuel: float | None = None

    # ---- ingest -------------------------------------------------------
    def update(self, f: Frame) -> NewLap | None:
        self.car = f
        self.frames.append(f)
        if not f.on_track:
            return None
        event = None
        if self._lap_frames and f.lap > self._lap_frames[-1].lap and f.lap > 1:
            done = self._lap_frames
            time_ms = f.last_lap_ms
            fuel_used = max(0.0, (self._lap_start_fuel or done[0].fuel) - done[-1].fuel)
            rec = LapRecord(done[-1].lap, time_ms, fuel_used, done)
            self.laps.append(rec)
            event = NewLap(rec)
            self._lap_frames = []
            self._lap_start_fuel = None
        if self._lap_start_fuel is None:
            self._lap_start_fuel = f.fuel
        self._lap_frames.append(f)
        return event

    def add_report(self, kind: str, value: str, t: float | None = None) -> None:
        self.reports.append(DriverReport(t if t is not None else self.car.t if self.car else 0.0, kind, value))

    # ---- queries ------------------------------------------------------
    @property
    def lap(self) -> int:
        return self.car.lap if self.car else 0

    @property
    def current_lap_frames(self) -> list[Frame]:
        return self._lap_frames

    def recent(self, seconds: float) -> list[Frame]:
        """Frames from the last `seconds` of game-clock history."""
        if not self.frames:
            return []
        cutoff = self.frames[-1].t - seconds
        out = []
        for f in reversed(self.frames):
            if f.t < cutoff:
                break
            out.append(f)
        out.reverse()
        return out

    def latest_report(self, kind: str) -> DriverReport | None:
        for r in reversed(self.reports):
            if r.kind == kind:
                return r
        return None

    @property
    def best_lap_ms(self) -> int:
        return self.car.best_lap_ms if self.car else -1

    @property
    def fuel_per_lap(self) -> float | None:
        """Mean fuel burn over the last few completed laps."""
        laps = [l.fuel_used for l in list(self.laps)[-3:] if l.fuel_used > 0]
        return sum(laps) / len(laps) if laps else None

    @property
    def laps_remaining(self) -> int | None:
        if not self.car or self.car.total_laps <= 0:
            return None
        return max(0, self.car.total_laps - self.car.lap + 1)

    def summary(self) -> dict:
        """Compact dict for prompts and CLI readouts."""
        c = self.car
        if c is None:
            return {"status": "no telemetry"}
        fpl = self.fuel_per_lap
        out = {
            "lap": c.lap,
            "total_laps": c.total_laps or None,
            "speed_kph": round(c.speed_kph),
            "gear": c.gear,
            "fuel_l": round(c.fuel, 1),
            "fuel_pct": round(100 * c.fuel / c.fuel_capacity) if c.fuel_capacity else None,
            "fuel_per_lap_l": round(fpl, 2) if fpl else None,
            "best_lap_ms": c.best_lap_ms if c.best_lap_ms > 0 else None,
            "last_lap_ms": c.last_lap_ms if c.last_lap_ms > 0 else None,
            "tyre_temp_c": [round(x) for x in c.tyre_temp],
        }
        if fpl and fpl > 0:
            out["laps_of_fuel"] = round(c.fuel / fpl, 1)
        wear = self.latest_report("tyre_wear")
        if wear:
            out["driver_reported_tyre_wear"] = wear.value
        return out
