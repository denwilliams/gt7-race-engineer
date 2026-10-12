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
) -> AsyncIterator[Frame]:
    """Yield frames for a short race. Lap times wobble so pace agents have
    something to chew on; fuel burn is steady with the pace."""
    dt = 1.0 / TICK_HZ
    t = 0.0
    packet_id = 0
    fuel = start_fuel
    best, last = -1, -1
    for lap in range(1, total_laps + 1):
        this_lap = lap_seconds * (1.0 + 0.04 * math.sin(lap * 1.7)) / speedup
        ticks = int(this_lap * TICK_HZ)
        for i in range(ticks):
            phase = i / ticks
            corner = math.sin(phase * 2 * math.pi * 4)
            speed = 55 + 25 * corner  # m/s
            fuel -= fuel_per_lap / ticks
            packet_id += 1
            yield Frame(
                t=t,
                packet_id=packet_id,
                position=(100 * math.cos(phase * 6.283), 0.0, 100 * math.sin(phase * 6.283)),
                speed=speed,
                rpm=4000 + 3000 * (corner + 1) / 2,
                fuel=max(fuel, 0.0),
                fuel_capacity=start_fuel,
                boost=0.0,
                oil_temp=95.0,
                water_temp=88.0,
                tyre_temp=(80 + 10 * phase, 82 + 10 * phase, 78 + 8 * phase, 79 + 8 * phase),
                lap=lap,
                total_laps=total_laps,
                best_lap_ms=best,
                last_lap_ms=last,
                start_position=3,
                num_cars=12,
                flags=FLAG_ON_TRACK,
                gear=3 + int(2 * (corner + 1) / 2),
                throttle=max(0.0, corner),
                brake=max(0.0, -corner),
            )
            t += dt
            if realtime:
                await asyncio.sleep(dt)
        last = int(this_lap * speedup * 1000)
        best = last if best < 0 else min(best, last)
