"""Pit stop detection from telemetry alone.

A stop can be fuel only, tyres only, both, or neither, so the two services are detected independently:

- refuel: the car is stationary and the fuel level climbs (a real stop adds tens of litres in seconds)
- new tyres: the four tyre temperatures snap together to the same value and drop sharply

Either one makes a PitStop, emitted once the car rolls away again. A visit with no service leaves no
signal in the telemetry, so it is not reported (there is nothing to say about it).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .telemetry.packet import Frame

STILL_MS = 2.0  # m/s: below this the car counts as stationary
ROLLING_MS = 5.0  # m/s: above this the car has left the box
REFUEL_MIN_L = 3.0  # a climb smaller than this is sensor noise, not a refuel
REFUEL_QUIET_S = 1.5  # fuel not rising for this long ends a refuel
TYRE_DROP_C = 5.0  # fresh tyres start well below worked tyres (observed: ~72 C -> 60.0 C)
TYRE_EVEN_C = 0.6  # new tyres read the same on all four corners
TYRE_UNEVEN_C = 1.5  # worked tyres differ by more than this
TYRE_WINDOW_S = 8.0
TYRE_COOLDOWN_S = 60.0
SERVICE_TIMEOUT_S = 25.0  # a service not followed by driving away within this is discarded


@dataclass(slots=True)
class TyreChange:
    lap: int
    t: float


@dataclass
class PitStop:
    lap: int  # lap the service happened on
    t: float  # game-receive time when the car rolled away
    added_l: float | None  # litres added, None if no refuel
    tyres_changed: bool
    fuel_after: float
    announced: bool = False  # set once it has been spoken (the lap update may include it)


class PitDetector:
    def __init__(self) -> None:
        self._prev: Frame | None = None
        self._rf_start: float | None = None  # fuel level when the current refuel began
        self._rf_last_t = 0.0
        self._rf_last_fuel = 0.0
        self._temps: deque[tuple[float, float, float]] = deque()  # (t, mean, spread)
        self._last_tyre_t = float("-inf")
        self._service: dict | None = None

    def update(self, f: Frame) -> list[TyreChange | PitStop]:
        events: list[TyreChange | PitStop] = []
        prev, self._prev = self._prev, f
        still = f.speed < STILL_MS

        # ---- refuel: stationary and the tank filling
        if prev is not None and still and f.fuel > prev.fuel + 0.05:
            if self._rf_start is None:
                self._rf_start = prev.fuel
            self._rf_last_t, self._rf_last_fuel = f.t, f.fuel
        elif self._rf_start is not None and (f.t - self._rf_last_t > REFUEL_QUIET_S or not still):
            added = self._rf_last_fuel - self._rf_start
            self._rf_start = None
            if added >= REFUEL_MIN_L:
                self._serve(f, added=added)

        # ---- new tyres: all four snap to the same, much lower temperature
        mean = sum(f.tyre_temp) / 4
        spread = max(f.tyre_temp) - min(f.tyre_temp)
        self._temps.append((f.t, mean, spread))
        while self._temps and f.t - self._temps[0][0] > TYRE_WINDOW_S:
            self._temps.popleft()
        if f.speed < ROLLING_MS and spread < TYRE_EVEN_C and f.t - self._last_tyre_t > TYRE_COOLDOWN_S:
            uneven = [m for _, m, s in self._temps if s > TYRE_UNEVEN_C]
            if uneven and max(uneven) - mean >= TYRE_DROP_C:
                self._last_tyre_t = f.t
                events.append(TyreChange(f.lap, f.t))
                self._serve(f, tyres=True)

        # ---- the car drives away: the stop is complete
        if self._service is not None and self._rf_start is None:
            if f.t - self._service["t"] > SERVICE_TIMEOUT_S:
                self._service = None
            elif f.speed > ROLLING_MS:
                s, self._service = self._service, None
                events.append(PitStop(s["lap"], f.t, s["added"], s["tyres"], f.fuel))
        return events

    def _serve(self, f: Frame, added: float | None = None, tyres: bool = False) -> None:
        s = self._service or {"lap": f.lap, "added": None, "tyres": False}
        s["t"] = f.t
        if added:
            s["added"] = (s["added"] or 0.0) + added
        if tyres:
            s["tyres"] = True
        self._service = s
