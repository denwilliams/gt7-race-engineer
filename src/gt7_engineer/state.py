"""Live car/race state with a rolling frame history that can be queried back in time."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .telemetry.packet import Frame
from .timefmt import fmt_lap

FRAME_HZ = 60
HISTORY_SECONDS = 600
REFUEL_JUMP_L = 1.0  # a rise above the lowest recent level by more than this is a pit stop; smaller blips are sensor noise (real GT7 jitters up to ~0.4 L)
POSITION_STALE_S = 5.0  # a position not refreshed for this long is no longer 'current'
REGRESS_FRAMES = 10  # the lap number must stay below its peak this many frames in a row to count as a new race
MIN_ESTIMATE_S = 20.0  # observed running time needed before burn-rate extrapolation is trusted
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
class NewRace:
    """Emitted by StateStore.update() when the state was reset because a new race began."""

    reason: str


@dataclass(slots=True)
class NewLap:
    """Event emitted by StateStore.update() when a lap completes."""

    record: LapRecord


class StateStore:
    def __init__(self, history_seconds: int = HISTORY_SECONDS, keep_laps: int = 10):
        self._args = (history_seconds, keep_laps)
        self._max_lap = 0
        self._regress = 0
        self.frames: deque[Frame] = deque(maxlen=history_seconds * FRAME_HZ)
        self.laps: deque[LapRecord] = deque(maxlen=keep_laps)
        self.reports: list[DriverReport] = []
        self.car: Frame | None = None
        self._lap_frames: list[Frame] = []
        self._lap_start_fuel: float | None = None
        self._refuelled = False
        self._lap_partial: bool | None = None  # None until the first lap is seen; True if we joined it mid-lap
        self._prev_fuel: float | None = None
        self._lap_min_fuel = float("inf")  # lowest fuel level seen this lap (to tell refuels from jitter)
        self._fuel_burned = 0.0  # litres burnt this race: every drop in the tank counts, refuels don't subtract
        self._observed_s = 0.0  # running time we have actually watched (for the burn-rate estimate)
        self._prev_t: float | None = None
        self.fuel_at_lap_start: dict[int, float] = {}  # lap number -> litres when it began (whole race)
        self._position: int | None = None  # last position reported by telemetry
        self._position_t = 0.0
        self.num_cars: int | None = None
        self.start_position: int | None = None  # first position seen, on the grid / lap 1
        self.position_at_lap_start: dict[int, int] = {}

    # ---- ingest -------------------------------------------------------
    def reset(self) -> None:
        """Forget everything about the current race (laps, fuel, positions, driver reports, history)."""
        self.__init__(*self._args)

    def _detect_new_race(self, f: Frame) -> NewRace | None:
        """A race restart shows up as the lap counter dropping back (e.g. 6 -> 0 or 1) and staying there."""
        if not f.on_track:  # paused/loading frames carry no race information
            return None
        if f.lap >= self._max_lap:
            if self._max_lap == 0 or f.lap <= self._max_lap + 1:  # a jump of more than one lap is a glitch
                self._max_lap = f.lap
            self._regress = 0
            return None
        self._regress += 1
        if self._regress < REGRESS_FRAMES:
            return None
        reason = f"lap counter went from {self._max_lap} back to {f.lap}"
        self.reset()
        self._max_lap = f.lap
        return NewRace(reason)

    def update(self, f: Frame) -> NewLap | NewRace | None:
        race = self._detect_new_race(f)
        event = self._ingest(f)
        return race or event

    def _ingest(self, f: Frame) -> NewLap | None:
        self.car = f
        self.frames.append(f)
        if f.race_position > 0:  # -1 in modes that don't report it
            self._position, self._position_t = f.race_position, f.t
            if f.num_cars > 0:
                self.num_cars = f.num_cars
            if self.start_position is None and f.lap <= 1:
                self.start_position = f.race_position
        if not f.on_track:
            return None
        if self._prev_t is not None:
            self._observed_s += min(1.0, max(0.0, f.t - self._prev_t))  # gaps in the stream don't count as time
        self._prev_t = f.t
        if self._prev_fuel is None or f.fuel - self._prev_fuel >= REFUEL_JUMP_L:
            self._prev_fuel = f.fuel  # first reading, or a real refuel: new baseline
        elif f.fuel < self._prev_fuel:
            self._fuel_burned += self._prev_fuel - f.fuel
            self._prev_fuel = f.fuel
        # a small rise is jitter: keep the lower baseline so the same litres are not counted twice
        if self._lap_frames and not (self._lap_frames[-1].lap <= f.lap <= self._lap_frames[-1].lap + 1):
            return None  # lap numbers only ever go up by one: anything else is a glitch (or a new race, handled before)
        event = None
        if self._lap_frames and f.lap > self._lap_frames[-1].lap and f.lap > 1:
            done = self._lap_frames
            time_ms = f.last_lap_ms
            fuel_used = max(0.0, (self._lap_start_fuel or done[0].fuel) - done[-1].fuel)
            rec = LapRecord(
                done[-1].lap, time_ms, fuel_used, done, burn_valid=not self._refuelled and not self._lap_partial,
                fuel_start=self._lap_start_fuel or done[0].fuel,
            )
            self.laps.append(rec)
            event = NewLap(rec)
            self._lap_frames = []
            self._lap_start_fuel = None
            self._lap_min_fuel = float("inf")
            self._refuelled = False
            self._lap_partial = False  # every lap after a line crossing we saw is complete
        if self._lap_frames and f.fuel > self._lap_min_fuel + REFUEL_JUMP_L:
            self._refuelled = True
        self._lap_min_fuel = min(self._lap_min_fuel, f.fuel)
        if self._lap_start_fuel is None:
            self._lap_start_fuel = f.fuel
            if self._lap_partial is None:  # first lap we see: complete only if it starts from the grid
                self._lap_partial = not (f.lap <= 1 and f.speed < 2.0)
        if f.lap >= 1 and not self._lap_partial and f.lap not in self.fuel_at_lap_start:
            self.fuel_at_lap_start[f.lap] = f.fuel  # first frame seen with this lap number
            if self.position is not None:
                self.position_at_lap_start[f.lap] = self.position
        self._lap_frames.append(f)
        return event

    def fuel_at_start_of(self, lap: int) -> float | None:
        """Litres in the tank when `lap` began (None if that lap hasn't started or wasn't seen)."""
        return self.fuel_at_lap_start.get(lap)

    def add_report(self, kind: str, value: str, t: float | None = None) -> None:
        self.reports.append(DriverReport(t if t is not None else self.car.t if self.car else 0.0, kind, value))

    # ---- queries ------------------------------------------------------
    @property
    def position(self) -> int | None:
        """Current race position from telemetry; None if the mode doesn't report it (or it went stale)."""
        if self._position is None or self.car is None or self.car.t - self._position_t > POSITION_STALE_S:
            return None
        return self._position

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
        """Mean fuel burn over the last few complete laps, or an estimate until one is available."""
        b = self.burns[-3:]
        return sum(b) / len(b) if b else self.estimated_fuel_per_lap

    @property
    def fuel_per_lap_is_estimate(self) -> bool:
        return not self.burns and self.estimated_fuel_per_lap is not None

    @property
    def estimated_fuel_per_lap(self) -> float | None:
        """Burn rate (litres per second watched) times the last lap time.

        Lets the engineer talk about fuel straight away after joining mid-race, before a
        complete lap has been seen. Needs a last lap time and some watched burn.
        """
        c = self.car
        if c is None or c.last_lap_ms <= 0 or self._observed_s < MIN_ESTIMATE_S or self._fuel_burned < 0.5:
            return None
        return self._fuel_burned / self._observed_s * c.last_lap_ms / 1000

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
            "fuel_per_lap_is_estimate": self.fuel_per_lap_is_estimate or None,
            # formatted as m:ss.mmm so the model reads them instead of converting raw milliseconds
            "last_lap_time": fmt_lap(c.last_lap_ms) if c.last_lap_ms > 0 else None,
            "best_lap_time": fmt_lap(c.best_lap_ms) if c.best_lap_ms > 0 else None,
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
        if c.last_lap_ms > 0 and c.best_lap_ms > 0:
            out["last_lap_vs_best_s"] = round((c.last_lap_ms - c.best_lap_ms) / 1000, 2)  # + slower, - faster
        if self.fuel_at_lap_start:
            out["fuel_at_race_start_l"] = round(self.fuel_at_lap_start[min(self.fuel_at_lap_start)], 1)
        if self._fuel_burned > 0:
            out["fuel_burned_total_l"] = round(self._fuel_burned, 1)
        if self._lap_start_fuel is not None and not self._refuelled and not self._lap_partial:
            out["fuel_used_this_lap_so_far_l"] = round(max(0.0, self._lap_start_fuel - c.fuel), 1)
        if self.fuel_at_lap_start:
            out["fuel_at_lap_start_l"] = {lap: round(v, 1) for lap, v in sorted(self.fuel_at_lap_start.items())}
        if self.position is not None:
            out["position"] = self.position
        if self.num_cars:
            out["cars_in_race"] = self.num_cars
        if self.start_position:
            out["starting_position"] = self.start_position
        if self.position_at_lap_start:
            out["position_at_lap_start"] = dict(sorted(self.position_at_lap_start.items()))
        pos = self.latest_report("position")
        if self.position is None:  # the mode doesn't report it: fall back to what the driver said
            out["position_note"] = (
                f"not in telemetry; driver last said: {pos.value}" if pos else "unknown (not in telemetry; ask the driver)"
            )
        wear = self.latest_report("tyre_wear")
        if wear:
            out["driver_reported_tyre_wear"] = wear.value
        return out
