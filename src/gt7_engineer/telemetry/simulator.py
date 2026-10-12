"""Fake GT7 session so everything can run without a PlayStation."""

from __future__ import annotations

import asyncio
import math
from collections.abc import AsyncIterator

from .packet import FLAG_ON_TRACK, Frame

TICK_HZ = 60.0


async def frames(
    lap_seconds: float = 30.0,
    total_laps: int = 5,
    speedup: float = 1.0,
    fuel_per_lap: float = 7.0,
    start_fuel: float = 40.0,
    realtime: bool = True,
    burn_growth: float = 0.0,
    refuel_lap: int | None = None,
    grid_position: int = 8,
    position_mode: str = "live",
    pit: dict | None = None,
) -> AsyncIterator[Frame]:
    """Yield frames for a short race. Lap times wobble so pace agents have
    something to chew on. Fuel burn is steady unless burn_growth (fractional
    increase per lap) is set; refuel_lap adds a mid-lap refuel to full. Position
    starts at grid_position and gains one place per lap ("live", what a real PS5
    reports); "prerace" reports it only before the start, as the GT7 docs say.
    pit={"lap": 3, "refuel": 40.0, "tyres": True, "stop_s": 6} makes a stop mid-lap: the car sits
    still while fuel is added and/or the tyres are changed (either may be omitted)."""
    dt = 1.0 / TICK_HZ
    t = 0.0
    packet_id = 0
    fuel = start_fuel
    best, last = -1, -1
    tyres_changed = False  # after new tyres the temperatures restart low (and stay uneven, like worked tyres)

    def tyre_temps(phase: float, fresh: bool) -> tuple[float, float, float, float]:
        if fresh:  # new tyres read identical on all four corners at first (60.0 in a real race), then drift apart
            b, drift = 60 + 10 * (phase - 0.5), max(0.0, phase - 0.5) * 2
            return (b, b + 0.7 * drift, b + 2.5 * drift, b + 2.0 * drift)
        return (80 + 10 * phase, 82 + 10 * phase, 78 + 8 * phase, 79 + 8 * phase)

    def make(lap, phase, corner, speed, fuel_v, tyres, t_v, pid, best_v, last_v):
        return Frame(
            t=t_v,
            packet_id=pid,
            position=(100 * math.cos(phase * 6.283), 0.0, 100 * math.sin(phase * 6.283)),
            speed=speed,
            rpm=4000 + 3000 * (corner + 1) / 2,
            fuel=max(fuel_v, 0.0),
            fuel_capacity=start_fuel,
            boost=0.0,
            oil_temp=95.0,
            water_temp=88.0,
            tyre_temp=tyres,
            lap=lap,
            total_laps=total_laps,
            best_lap_ms=best_v,
            last_lap_ms=last_v,
            race_position=(
                max(1, grid_position - (lap - 1)) if position_mode == "live" else (grid_position if pid <= 30 else -1)
            ),
            num_cars=12 if position_mode == "live" or pid <= 30 else -1,
            flags=FLAG_ON_TRACK,
            gear=3 + int(2 * (corner + 1) / 2),
            throttle=max(0.0, corner),
            brake=max(0.0, -corner),
        )

    for lap in range(1, total_laps + 1):
        this_lap = lap_seconds * (1.0 + 0.04 * math.sin(lap * 1.7)) / speedup
        ticks = int(this_lap * TICK_HZ)
        lap_burn = fuel_per_lap * (1.0 + burn_growth) ** (lap - 1)
        for i in range(ticks):
            phase = i / ticks
            corner = math.sin(phase * 2 * math.pi * 4)
            speed = 55 + 25 * corner  # m/s
            if lap == 1:  # standing start: sit on the grid for 1 s, then ramp up over 2 s
                speed *= max(0.0, min(1.0, (i - TICK_HZ) / (2 * TICK_HZ)))
            fuel -= lap_burn / ticks
            if lap == refuel_lap and i == ticks // 2:
                fuel = start_fuel
            if pit and lap == pit["lap"] and i == ticks // 2:
                # A pit stop: the car sits still while the tank fills and/or the tyres are swapped.
                stop_ticks = int(pit.get("stop_s", 6.0) * TICK_HZ)
                add = pit.get("refuel") or 0.0
                before = tyre_temps(phase, tyres_changed)
                after = tyre_temps(phase, tyres_changed or bool(pit.get("tyres")))
                for k in range(stop_ticks):
                    mix = min(1.0, k / TICK_HZ)  # tyres snap to their new temperature over one second
                    tyres = tuple(b + (a_ - b) * mix for a_, b in zip(after, before))
                    fuel_now = fuel + add * min(1.0, k / (4 * TICK_HZ))  # the tank fills over four seconds
                    packet_id += 1
                    yield make(lap, phase, corner, 0.0, fuel_now, tyres, t, packet_id, best, last)
                    t += dt
                    if realtime:
                        await asyncio.sleep(dt)
                fuel += add
                tyres_changed = tyres_changed or bool(pit.get("tyres"))
            packet_id += 1
            yield make(lap, phase, corner, speed, fuel, tyre_temps(phase, tyres_changed), t, packet_id, best, last)
            t += dt
            if realtime:
                await asyncio.sleep(dt)
        last = int(this_lap * speedup * 1000)
        best = last if best < 0 else min(best, last)
