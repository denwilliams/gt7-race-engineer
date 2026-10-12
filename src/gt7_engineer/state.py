"""Live car/race state with a rolling frame history that can be queried back in time."""

from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass, field

from .pit import PitDetector, PitStop, TyreChange
from .telemetry.packet import Frame
from .timefmt import fmt_lap
from .track import TrackModel

FRAME_HZ = 60
HISTORY_SECONDS = 600
REFUEL_JUMP_L = 1.0  # a rise above the lowest recent level by more than this is a pit stop; smaller blips are sensor noise (real GT7 jitters up to ~0.4 L)
POSITION_STALE_S = 5.0  # a position not refreshed for this long is no longer 'current'
REGRESS_FRAMES = 10  # the lap number must stay below its peak this many frames in a row to count as a new race
MIN_ESTIMATE_S = 20.0  # observed running time needed before burn-rate extrapolation is trusted
TREND_STEADY_PCT = 3.0  # burn changes smaller than this per lap count as steady


def _game_dt(a: Frame, b: Frame) -> float:
    """Game time between two frames, from GT7's own packet counter (60 per second).

    Receive time is wrong after packet loss or a stall: 12 lost packets arrive as one big step in
    almost no wall time. Falls back to receive time if the counter looks unusable.
    """
    n = b.packet_id - a.packet_id
    if 0 < n <= 600:
        return n / 60.0
    return max(0.0, b.t - a.t)


@dataclass(slots=True)
class LapRecord:
    lap: int
    time_ms: int  # as reported by the game; -1 if unknown
    fuel_used: float
    frames: list[Frame] = field(repr=False, default_factory=list)
    burn_valid: bool = True  # False if the car refuelled during the lap (pit stop)
    fuel_start: float = 0.0  # fuel level (litres) when the lap began
    distance_m: float = 0.0  # path length driven this lap (from position changes)
    distance_ok: bool = False  # False if we joined mid-lap or the car was teleported (pit box, respawn)


@dataclass(slots=True)
class DriverReport:
    t: float
    kind: str  # e.g. "tyre_wear"
    value: str  # free text, e.g. "fronts 60%, rears 70%"


@dataclass(slots=True)
class TrackMapped:
    """The turn map was (re)built or its turn count changed."""

    turns: int
    laps: int


@dataclass(slots=True)
class NewRace:
    """Emitted by StateStore.update() when the state was reset because a new race began."""

    reason: str


@dataclass(slots=True)
class NewLap:
    """Event emitted by StateStore.update() when a lap completes."""

    record: LapRecord


class StateStore:
    def __init__(self, history_seconds: int = HISTORY_SECONDS, keep_laps: int = 10, mirror: bool = False):
        self._args = (history_seconds, keep_laps, mirror)
        self.track = TrackModel(mirror=mirror)  # turns learned from the clean laps driven
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
        self.pit = PitDetector()
        self.pit_stops: list[PitStop] = []
        self.events: deque = deque(maxlen=50)  # drained by the app (pit stops, track map changes)
        self.tyres_first_lap: int | None = None  # first lap run on the current set of tyres (None: unknown)
        self._lap_dist = 0.0  # metres driven so far this lap
        self._lap_dist_ok = True  # False once a teleport is seen this lap
        self._race_start_fuel: float | None = None  # fuel on the grid, before the race's first line crossing
        self._joined: tuple[int, float] | None = None  # (lap, fuel) of the first frame we saw
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
        if self._joined is None:
            self._joined = (f.lap, f.fuel)
        if self._prev_t is not None:
            self._observed_s += min(1.0, max(0.0, f.t - self._prev_t))  # gaps in the stream don't count as time
        self._prev_t = f.t
        if self._prev_fuel is None or f.fuel - self._prev_fuel >= REFUEL_JUMP_L:
            self._prev_fuel = f.fuel  # first reading, or a real refuel: new baseline
        elif f.fuel < self._prev_fuel:
            self._fuel_burned += self._prev_fuel - f.fuel
            self._prev_fuel = f.fuel
        # a small rise is jitter: keep the lower baseline so the same litres are not counted twice
        for ev in self.pit.update(f):  # refuel / new tyres / pit stop complete
            if isinstance(ev, TyreChange):
                self.tyres_first_lap = ev.lap + 1  # the lap in progress is a part-lap on the new set
                self.reports = [r for r in self.reports if r.kind != "tyre_wear"]  # an old wear report is now wrong
            else:
                self.pit_stops.append(ev)
                self.events.append(ev)
        if self._lap_frames and not (self._lap_frames[-1].lap <= f.lap <= self._lap_frames[-1].lap + 1):
            return None  # lap numbers only ever go up by one: anything else is a glitch (or a new race, handled before)
        step, jumped = 0.0, False
        if self._lap_frames:  # distance since the previous frame, ignoring teleports (pit box, respawn)
            prev = self._lap_frames[-1]
            step = math.dist(prev.position, f.position)
            expected = prev.speed * _game_dt(prev, f)
            if step > max(5.0, 3 * expected + 3.0):
                step, jumped = expected, True
        event = None
        if self._lap_frames and f.lap > self._lap_frames[-1].lap and f.lap > 1:
            done = self._lap_frames
            time_ms = f.last_lap_ms
            fuel_used = max(0.0, (self._lap_start_fuel or done[0].fuel) - done[-1].fuel)
            rec = LapRecord(
                done[-1].lap, time_ms, fuel_used, done, burn_valid=not self._refuelled and not self._lap_partial,
                fuel_start=self._lap_start_fuel or done[0].fuel,
                distance_m=self._lap_dist, distance_ok=self._lap_dist_ok and not self._lap_partial,
            )
            self.laps.append(rec)
            if rec.distance_ok and len(done) > 100:  # a clean, complete lap: learn the turns from it
                before = len(self.track.turns)
                self.track.add_lap(rec.lap, done)
                if len(self.track.turns) != before:
                    self.events.append(TrackMapped(len(self.track.turns), len(self.track.profiles)))
            event = NewLap(rec)
            self._lap_frames = []
            self._lap_start_fuel = None
            self._lap_min_fuel = float("inf")
            self._refuelled = False
            self._lap_partial = False  # every lap after a line crossing we saw is complete
            self._lap_dist, self._lap_dist_ok = 0.0, True
        elif self._lap_frames and f.lap > self._lap_frames[-1].lap:
            # Lap 0 -> 1: the race itself starts at the line. What came before was the run-up, not a lap.
            if not self._lap_partial:
                self._race_start_fuel = self._lap_start_fuel
                self.tyres_first_lap = 1  # the race starts on a fresh set
            self._lap_frames = []
            self._lap_start_fuel = None
            self._lap_min_fuel = float("inf")
            self._refuelled = False
            self._lap_partial = False
            self._lap_dist, self._lap_dist_ok = 0.0, True
            step, jumped = 0.0, False
        self._lap_dist += step
        if jumped:
            self._lap_dist_ok = False
        if self._lap_frames and f.fuel > self._lap_min_fuel + REFUEL_JUMP_L:
            self._refuelled = True
        self._lap_min_fuel = min(self._lap_min_fuel, f.fuel)
        if self._lap_start_fuel is None:
            self._lap_start_fuel = f.fuel
            if self._lap_partial is None:  # first lap we see: complete only if it starts from the grid
                self._lap_partial = not (f.lap == 0 or (f.lap == 1 and f.speed < 2.0))  # grid/run-up, or a standing start
                if not self._lap_partial:
                    self.tyres_first_lap = 1  # we saw the race start: the tyres are the starting set
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
    def tyre_age_laps(self) -> int | None:
        """Complete laps run on the current set of tyres; None if the start of the set was not seen."""
        if self.tyres_first_lap is None or self.car is None:
            return None
        return max(0, self.car.lap - self.tyres_first_lap)

    @property
    def lap_length_m(self) -> float | None:
        """Typical lap length: the median of the clean complete laps (pit/teleport laps don't skew it)."""
        good = [l.distance_m for l in self.laps if l.distance_ok and l.distance_m > 100]
        return statistics.median(good) if good else None

    @property
    def lap_distance_so_far_m(self) -> float | None:
        """Metres driven on the current lap; None if we don't know where the lap began."""
        return self._lap_dist if self.car is not None and self.car.lap >= 1 and not self._lap_partial and self._lap_dist_ok else None

    @property
    def next_turn(self) -> dict | None:
        """The next turn ahead of the car and how far away turn-in is (needs the track mapped)."""
        d = self.lap_distance_so_far_m
        nxt = self.track.next_turn(d) if d is not None else None
        if nxt is None:
            return None
        turn, dist = nxt
        return {"number": turn.number, "direction": turn.label(self.track.mirror), "in_m": round(dist),
                "angle_deg": round(abs(turn.angle_deg))}

    @property
    def lap_progress(self) -> float | None:
        """Fraction (0..1) of the current lap driven, by distance. Approximate: needs one clean complete lap."""
        length, done = self.lap_length_m, self.lap_distance_so_far_m
        return None if not length or done is None else max(0.0, min(1.0, done / length))

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
        """Typical fuel burn: the median of the last few complete laps (one odd lap can't move it),
        or an estimate until a complete lap has been seen."""
        b = self.burns
        return statistics.median(b) if b else self.estimated_fuel_per_lap

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

        Theil-Sen slope (the median of the slopes between every pair of laps) over the last up to 5
        valid laps, so one wild lap cannot tip it; needs at least 3.
        """
        b = self.burns
        if len(b) < 3:
            return None
        slope = statistics.median((b[j] - b[i]) / (j - i) for i in range(len(b)) for j in range(i + 1, len(b)))
        ym = statistics.median(b)
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
        # Fuel at the race start: seen on the grid, else at the first line crossing, else assume a full tank.
        start = None
        assumed = False
        if self._race_start_fuel is not None:
            start = self._race_start_fuel
        elif 1 in self.fuel_at_lap_start:
            start = self.fuel_at_lap_start[1]
        elif self._joined is not None and c.fuel_capacity > 0:
            start, assumed = c.fuel_capacity, True
        if start is not None:
            out["fuel_at_race_start_l"] = round(start, 1)
            if assumed:
                out["fuel_at_race_start_note"] = (
                    f"assumed a full tank: the race start was not seen (tracking began on lap {self._joined[0]})"
                )
        if self._fuel_burned > 0 or assumed:
            burned = self._fuel_burned
            if assumed:  # add what was burned before we started watching
                burned += max(0.0, start - self._joined[1])
            out["fuel_burned_total_l"] = round(burned, 1)
        if self._lap_start_fuel is not None and not self._refuelled and not self._lap_partial:
            out["fuel_used_this_lap_so_far_l"] = round(max(0.0, self._lap_start_fuel - c.fuel), 1)
        if self.fuel_at_lap_start:
            out["fuel_at_lap_start_l"] = {lap: round(v, 1) for lap, v in sorted(self.fuel_at_lap_start.items())}
        if (age := self.tyre_age_laps) is not None:
            out["tyre_age_laps"] = age
        if self.pit_stops:
            out["pit_stops"] = [
                {"lap": p.lap, "fuel_added_l": round(p.added_l, 1) if p.added_l else None, "new_tyres": p.tyres_changed}
                for p in self.pit_stops
            ]
        if self.track.turns:
            m = self.track.mirror
            out["track_turns"] = [
                {"n": t.number, "dir": t.label(m), "apex_m": round(t.apex_s), "angle": round(abs(t.angle_deg))}
                for t in self.track.turns
            ]
            if (nt := self.next_turn) is not None:
                out["next_turn"] = nt
            if ins := self.track.insights()[:4]:
                out["turn_insights"] = [i.text for i in ins]
        if (length := self.lap_length_m) is not None:
            out["lap_length_m"] = round(length)
            if (prog := self.lap_progress) is not None:
                out["lap_progress_pct"] = round(100 * prog)
                out["lap_distance_so_far_m"] = round(self._lap_dist)
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
