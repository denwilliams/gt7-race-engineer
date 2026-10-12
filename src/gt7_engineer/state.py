"""Live car/race state with a rolling frame history that can be queried back in time."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .telemetry.packet import Frame

FRAME_HZ = 60
HISTORY_SECONDS = 600
REFUEL_JUMP_L = 0.5  # a rise in fuel level bigger than this between frames means a pit stop
TREND_STEADY_PCT = 3.0  # burn changes smaller than this per lap count as steady


@dataclass(slots=True)
class LapRecord:
    lap: int
    time_ms: int  # as reported by the game; -1 if unknown
    fuel_used: float
    frames: list[Frame] = field(repr=False, default_factory=list)
    burn_valid: bool = True  # False if the car refuelled during the lap (pit stop)
    fuel_start: float = 0.0  # fuel level (litres) when the lap began


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
        self._refuelled = False
        self.fuel_at_lap_start: dict[int, float] = {}  # lap number -> litres when it began (whole race)
        self.start_position: int | None = None  # grid position, only reported by GT7 before the race starts
        self.num_cars: int | None = None

    # ---- ingest -------------------------------------------------------
    def update(self, f: Frame) -> NewLap | None:
        self.car = f
        self.frames.append(f)
        if f.start_position > 0:  # GT7 reports -1 once the race has started, so remember it
            self.start_position, self.num_cars = f.start_position, f.num_cars if f.num_cars > 0 else self.num_cars
        if not f.on_track:
            return None
        event = None
        if self._lap_frames and f.lap > self._lap_frames[-1].lap and f.lap > 1:
            done = self._lap_frames
            time_ms = f.last_lap_ms
            fuel_used = max(0.0, (self._lap_start_fuel or done[0].fuel) - done[-1].fuel)
            rec = LapRecord(
                done[-1].lap, time_ms, fuel_used, done, burn_valid=not self._refuelled,
                fuel_start=self._lap_start_fuel or done[0].fuel,
            )
            self.laps.append(rec)
            event = NewLap(rec)
            self._lap_frames = []
            self._lap_start_fuel = None
            self._refuelled = False
        if self._lap_frames and f.fuel > self._lap_frames[-1].fuel + REFUEL_JUMP_L:
            self._refuelled = True
        if self._lap_start_fuel is None:
            self._lap_start_fuel = f.fuel
            if f.lap >= 1:
                self.fuel_at_lap_start[f.lap] = f.fuel
        self._lap_frames.append(f)
        return event

    def fuel_at_start_of(self, lap: int) -> float | None:
        """Litres in the tank when `lap` began (None if that lap hasn't started or wasn't seen)."""
        return self.fuel_at_lap_start.get(lap)

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
    def burns(self) -> list[float]:
        """Fuel burnt on each of the last 5 completed laps (litres), excluding pit-stop laps."""
        return [l.fuel_used for l in list(self.laps)[-5:] if l.burn_valid and l.fuel_used > 0]

    @property
    def last_lap_burn(self) -> float | None:
        b = self.burns
        return b[-1] if b else None

    @property
    def fuel_per_lap(self) -> float | None:
        """Mean fuel burn over the last few valid laps."""
        b = self.burns[-3:]
        return sum(b) / len(b) if b else None

    @property
    def burn_trend(self) -> tuple[float, str] | None:
        """(change in burn per lap in litres, 'heavier' | 'lighter' | 'steady').

        Least-squares slope over the last up to 5 valid laps; needs at least 3.
        """
        b = self.burns
        if len(b) < 3:
            return None
        n = len(b)
        xm, ym = (n - 1) / 2, sum(b) / n
        slope = sum((i - xm) * (y - ym) for i, y in enumerate(b)) / sum((i - xm) ** 2 for i in range(n))
        if abs(slope) / ym * 100 < TREND_STEADY_PCT:
            return slope, "steady"
        return slope, "heavier" if slope > 0 else "lighter"

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
        if (left := self.laps_remaining) is not None:
            out["laps_remaining_including_current"] = left
        if fpl and fpl > 0:
            out["laps_of_fuel"] = round(c.fuel / fpl, 1)
            if left is not None:
                margin = round(c.fuel / fpl - left, 1)
                out["fuel_margin_laps"] = margin  # negative: cannot finish without saving fuel or pitting
        out["fuel_burn_last_laps_l"] = [round(x, 2) for x in self.burns]
        if (trend := self.burn_trend) is not None:
            out["fuel_burn_trend"] = f"{trend[1]} ({trend[0]:+.2f} L/lap every lap)"
        if self.fuel_at_lap_start:
            out["fuel_at_lap_start_l"] = {lap: round(v, 1) for lap, v in sorted(self.fuel_at_lap_start.items())}
        if self.start_position:
            out["starting_position"] = self.start_position
            if self.num_cars:
                out["cars_in_race"] = self.num_cars
        pos = self.latest_report("position")
        out["current_position"] = (
            f"not in telemetry; driver last said: {pos.value}" if pos else "unknown (GT7 telemetry does not provide it; ask the driver)"
        )
        wear = self.latest_report("tyre_wear")
        if wear:
            out["driver_reported_tyre_wear"] = wear.value
        return out
