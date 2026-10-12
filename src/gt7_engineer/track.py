"""Learn the track from the laps driven: where the turns are, and what works best in each.

From one clean lap the car's path gives the turns (curvature of the driven line): how many, which way,
and how far from the start/finish line. Over several laps the turns that keep showing up form the map.
Each lap's pass through each turn is then measured (time through the zone, gear, braking point,
throttle pickup, speeds) so the fastest passes can be compared with the slowest ones.

Everything is derived from position, speed, throttle, brake and gear: nothing about the track is built in.
Distances are metres from the start/finish line. Approximate by nature (racing lines differ lap to lap).
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field

import numpy as np

from .telemetry.packet import Frame

STEP_M = 5.0  # resampling step along the lap
CURVE_SMOOTH_PTS = 7  # ~35 m moving average on the heading
CURVE_THRESHOLD = 1 / 250  # rad/m: tighter than a 250 m radius counts as turning
MIN_TURN_LEN_M = 25.0
MERGE_GAP_M = 30.0  # same-direction pieces closer than this are one turn
MIN_TURN_ANGLE_DEG = 20.0  # smaller wiggles are noise, not turns
CORE_FRACTION = 0.4  # turn-in/exit are where the curvature reaches this share of its peak (not the long gentle approach)
CLUSTER_TOL_M = 45.0  # the same turn on different laps lands within this
MIN_SHARE = 0.6  # a turn must be seen on this share of laps to count
ZONE_PAD_M = 100.0  # comparison zone: from this far before turn-in to this far after the exit
BRAKE_ON = 0.15
THROTTLE_ON = 0.8
MIN_GAIN_S = 0.15  # differences smaller than this are lap-to-lap noise
MIN_PASSES = 5


@dataclass(slots=True)
class Profile:
    """One lap resampled every STEP_M metres."""

    s: np.ndarray
    x: np.ndarray
    z: np.ndarray
    t: np.ndarray
    speed: np.ndarray  # m/s
    throttle: np.ndarray
    brake: np.ndarray
    gear: np.ndarray

    @property
    def length(self) -> float:
        return float(self.s[-1])

    def at(self, arr: np.ndarray, dist: float) -> float:
        return float(np.interp(dist, self.s, arr))


@dataclass(slots=True)
class Turn:
    start_s: float
    apex_s: float
    end_s: float
    angle_deg: float  # signed heading change
    sign: int  # +1 / -1: which way the heading rotates (which side is "left" depends on the game's axes)

    def label(self, mirror: bool = False) -> str:
        return "L" if (self.sign > 0) != mirror else "R"


@dataclass(slots=True)
class MapTurn:
    number: int
    start_s: float
    apex_s: float
    end_s: float
    angle_deg: float
    sign: int
    seen_on: int  # laps it was detected on

    def label(self, mirror: bool = False) -> str:
        return "L" if (self.sign > 0) != mirror else "R"


@dataclass(slots=True)
class TurnPass:
    """How one lap went through one turn."""

    lap: int
    zone_time: float  # seconds from ZONE_PAD_M before turn-in to ZONE_PAD_M after the exit
    entry_kph: float
    min_kph: float
    exit_kph: float
    gear: int  # gear at the slowest point
    brake_start_m: float | None  # metres before turn-in where the braking for it began; None: no braking
    brake_peak: float
    throttle_pickup_m: float | None  # metres after the slowest point until throttle > 80%


@dataclass(slots=True)
class Insight:
    turn: int
    label: str
    gap_s: float  # how much faster the fastest passes were than the slowest
    text: str


# ---------------------------------------------------------------- profile
def make_profile(frames: list[Frame], step: float = STEP_M) -> Profile | None:
    """Resample a lap by distance travelled. Returns None for laps too short to use."""
    if len(frames) < 100:
        return None
    pos = np.array([f.position for f in frames], dtype=float)
    x, z = pos[:, 0], pos[:, 2]
    d = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(z)))])
    if d[-1] < 500:
        return None
    s = np.arange(0.0, d[-1], step)
    tt = np.array([f.t for f in frames], dtype=float)
    gear = np.array([f.gear for f in frames], dtype=float)
    return Profile(
        s=s,
        x=np.interp(s, d, x),
        z=np.interp(s, d, z),
        t=np.interp(s, d, tt),
        speed=np.interp(s, d, [f.speed for f in frames]),
        throttle=np.interp(s, d, [f.throttle for f in frames]),
        brake=np.interp(s, d, [f.brake for f in frames]),
        gear=np.round(np.interp(s, d, gear)),
    )


def curvature(p: Profile) -> np.ndarray:
    """Signed rate of heading change (rad per metre) along the driven line."""
    heading = np.unwrap(np.arctan2(np.gradient(p.z), np.gradient(p.x)))
    k = CURVE_SMOOTH_PTS
    smooth = np.convolve(np.pad(heading, k // 2, mode="edge"), np.ones(k) / k, mode="valid")
    return np.gradient(smooth, p.s)


def detect_turns(p: Profile) -> list[Turn]:
    kappa = curvature(p)
    s = p.s
    on = np.abs(kappa) > CURVE_THRESHOLD
    segs: list[list[int]] = []
    i = 0
    while i < len(s):
        if not on[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(s) and on[j + 1] and np.sign(kappa[j + 1]) == np.sign(kappa[i]):
            j += 1
        if s[j] - s[i] >= MIN_TURN_LEN_M:
            segs.append([i, j])
        i = j + 1
    merged: list[list[int]] = []
    for a, b in segs:
        if merged and s[a] - s[merged[-1][1]] < MERGE_GAP_M and np.sign(kappa[a]) == np.sign(kappa[merged[-1][0]]):
            merged[-1][1] = b
        else:
            merged.append([a, b])
    turns = []
    for a, b in merged:
        angle = math.degrees(float(np.trapezoid(kappa[a : b + 1], s[a : b + 1])))
        if abs(angle) < MIN_TURN_ANGLE_DEG:
            continue
        mag = np.abs(kappa[a : b + 1])
        # Apex = where half of the turning is done. (The point of peak curvature is arbitrary on a
        # circular arc and wanders by tens of metres between laps.)
        cum = np.abs(np.cumsum(kappa[a : b + 1]))
        apex = a + int(np.searchsorted(cum, 0.5 * cum[-1]))
        core = np.nonzero(mag >= CORE_FRACTION * mag.max())[0]  # where it is really turning
        turns.append(Turn(float(s[a + core[0]]), float(s[apex]), float(s[a + core[-1]]), angle, 1 if angle > 0 else -1))
    return turns


# ---------------------------------------------------------------- consensus map
def consensus(per_lap: dict[int, list[Turn]]) -> list[MapTurn]:
    """The turns seen on most laps, matched by position along the lap."""
    n_laps = len(per_lap)
    if n_laps == 0:
        return []
    items = sorted(((t.apex_s, lap, t) for lap, ts in per_lap.items() for t in ts), key=lambda x: x[0])
    clusters: list[list[tuple[int, Turn]]] = []
    for apex, lap, turn in items:
        for c in reversed(clusters[-3:]):  # only the nearest few clusters can match
            mean = statistics.fmean(t.apex_s for _, t in c)
            if abs(apex - mean) < CLUSTER_TOL_M and turn.sign == c[0][1].sign and all(l != lap for l, _ in c):
                c.append((lap, turn))
                break
        else:
            clusters.append([(lap, turn)])
    need = max(1, math.ceil(MIN_SHARE * n_laps)) if n_laps > 2 else n_laps
    kept = [c for c in clusters if len(c) >= need]
    out = []
    for k, c in enumerate(sorted(kept, key=lambda c: statistics.median(t.apex_s for _, t in c)), 1):
        ts = [t for _, t in c]
        out.append(MapTurn(
            number=k,
            start_s=statistics.median(t.start_s for t in ts),
            apex_s=statistics.median(t.apex_s for t in ts),
            end_s=statistics.median(t.end_s for t in ts),
            angle_deg=statistics.median(t.angle_deg for t in ts),
            sign=ts[0].sign,
            seen_on=len(ts),
        ))
    return out


# ---------------------------------------------------------------- one lap through one turn
def measure_pass(p: Profile, turn: MapTurn, lap: int) -> TurnPass | None:
    s0, s1 = max(0.0, turn.start_s - ZONE_PAD_M), min(p.length, turn.end_s + ZONE_PAD_M)
    if s1 - s0 < 50:
        return None
    zone_time = p.at(p.t, s1) - p.at(p.t, s0)
    lo, hi = int(max(0, turn.start_s - 20) / STEP_M), int(min(p.length, turn.end_s + 20) / STEP_M) + 1
    i_min = lo + int(np.argmin(p.speed[lo:hi]))
    # braking for this turn: the last contiguous brake application that ends between 60 m before turn-in and the apex
    brake_on = p.brake > BRAKE_ON
    i_turn, i_apex = int(turn.start_s / STEP_M), int(turn.apex_s / STEP_M)
    brake_start = peak = None
    j = min(i_apex, len(brake_on) - 1)
    lowest = max(0, i_turn - int(60 / STEP_M))
    while j >= lowest and not brake_on[j]:
        j -= 1
    if j >= lowest:  # braking is happening at j; walk back to where it began
        end_j = j
        while j > 0 and brake_on[j - 1]:
            j -= 1
        if end_j - j >= 2:  # at least ~10 m: a real brake application, not a tap
            brake_start = turn.start_s - float(p.s[j])
            peak = float(p.brake[j : end_j + 1].max())
    pickup = None
    for k in range(i_min, min(len(p.s), i_min + int(150 / STEP_M))):
        if p.throttle[k] > THROTTLE_ON:
            pickup = float(p.s[k] - p.s[i_min])
            break
    return TurnPass(
        lap=lap, zone_time=float(zone_time),
        entry_kph=p.at(p.speed, turn.start_s) * 3.6, min_kph=float(p.speed[i_min]) * 3.6,
        exit_kph=p.at(p.speed, min(p.length, turn.end_s + 30)) * 3.6,
        gear=int(p.gear[i_min]), brake_start_m=brake_start, brake_peak=peak or 0.0, throttle_pickup_m=pickup,
    )


# ---------------------------------------------------------------- comparing laps
def drop_outliers(passes: list[TurnPass]) -> list[TurnPass]:
    """Remove passes whose time is far off the typical one (ran wide, stuck behind a car, a spin).

    Those laps are not a pattern to learn from. Uses the median and the median absolute deviation, so
    one wild lap cannot drag the reference along with it.
    """
    if len(passes) < 4:
        return passes
    times = [p.zone_time for p in passes]
    med = statistics.median(times)
    mad = statistics.median(abs(t - med) for t in times) * 1.4826
    limit = 3 * max(mad, 0.05)  # never tighter than 0.15 s: identical laps would otherwise reject everything
    return [p for p in passes if abs(p.zone_time - med) <= limit]


def compare_passes(turn: MapTurn, passes: list[TurnPass], mirror: bool = False) -> Insight | None:
    """What the fastest passes through this turn did differently from the slowest."""
    passes = drop_outliers(passes)
    if len(passes) < MIN_PASSES:
        return None
    ranked = sorted(passes, key=lambda p: p.zone_time)
    k = max(2, len(ranked) // 3)
    fast, slow = ranked[:k], ranked[-k:]
    gap = statistics.median(p.zone_time for p in slow) - statistics.median(p.zone_time for p in fast)
    if gap < MIN_GAIN_S:
        return None
    parts: list[str] = []
    g_fast = statistics.mode(p.gear for p in fast)
    g_slow = statistics.mode(p.gear for p in slow)
    if g_fast != g_slow:
        parts.append(f"used gear {g_fast} instead of {g_slow}")
    dv = statistics.median(p.min_kph for p in fast) - statistics.median(p.min_kph for p in slow)
    if abs(dv) >= 3:
        parts.append(f"carried {abs(dv):.0f} km/h {'more' if dv > 0 else 'less'} speed at the slowest point")
    bf = [p.brake_start_m for p in fast if p.brake_start_m is not None]
    bs = [p.brake_start_m for p in slow if p.brake_start_m is not None]
    if len(bf) >= k // 2 + 1 and len(bs) >= k // 2 + 1:
        db = statistics.median(bs) - statistics.median(bf)  # >0: the fast laps braked closer to the turn (later)
        if abs(db) >= 8:
            parts.append(f"braked {abs(db):.0f} m {'later' if db > 0 else 'earlier'}")
    elif len(bf) < k / 2 and len(bs) >= k // 2 + 1:
        parts.append("lifted instead of braking")
    tf = [p.throttle_pickup_m for p in fast if p.throttle_pickup_m is not None]
    ts = [p.throttle_pickup_m for p in slow if p.throttle_pickup_m is not None]
    if tf and ts:
        dt = statistics.median(ts) - statistics.median(tf)
        if abs(dt) >= 8:
            parts.append(f"got back on the throttle {abs(dt):.0f} m {'sooner' if dt > 0 else 'later'}")
    if not parts:
        return None
    label = turn.label(mirror)
    detail = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
    return Insight(turn.number, label, gap, f"Turn {turn.number} {label}: the fastest laps {detail}. Worth about {gap:.1f} s.")


# ---------------------------------------------------------------- the model the state store keeps
@dataclass
class TrackModel:
    mirror: bool = False  # flip the L/R labels if the game's axes make them come out reversed
    profiles: dict[int, Profile] = field(default_factory=dict)
    per_lap: dict[int, list[Turn]] = field(default_factory=dict)
    turns: list[MapTurn] = field(default_factory=list)
    passes: dict[int, dict[int, TurnPass]] = field(default_factory=dict)  # turn number -> lap -> pass

    @property
    def lap_length_m(self) -> float | None:
        return statistics.median(p.length for p in self.profiles.values()) if self.profiles else None

    def add_lap(self, lap: int, frames: list[Frame]) -> bool:
        """Learn from a clean, complete lap. Returns True if the turn map changed."""
        prof = make_profile(frames)
        if prof is None or lap in self.profiles:
            return False
        self.profiles[lap] = prof
        self.per_lap[lap] = detect_turns(prof)
        before = [(t.number, round(t.apex_s), t.sign) for t in self.turns]
        self.turns = consensus(self.per_lap)
        self.passes = {t.number: {} for t in self.turns}
        for lp, pf in self.profiles.items():  # the map may have shifted: re-measure every lap against it
            for t in self.turns:
                m = measure_pass(pf, t, lp)
                if m is not None:
                    self.passes[t.number][lp] = m
        return before != [(t.number, round(t.apex_s), t.sign) for t in self.turns]

    def next_turn(self, distance_m: float) -> tuple[MapTurn, float] | None:
        """The next turn ahead and how many metres until turn-in (wraps round the lap)."""
        if not self.turns:
            return None
        length = self.lap_length_m or self.turns[-1].end_s
        for t in self.turns:
            if t.start_s - distance_m > 0:
                return t, t.start_s - distance_m
        t = self.turns[0]
        return t, length - distance_m + t.start_s

    def insights(self) -> list[Insight]:
        out = []
        for t in self.turns:
            ins = compare_passes(t, list(self.passes.get(t.number, {}).values()), self.mirror)
            if ins is not None:
                out.append(ins)
        return sorted(out, key=lambda i: -i.gap_s)
