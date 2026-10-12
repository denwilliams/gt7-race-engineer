"""Record telemetry to a JSONL file so a real race can be analysed offline (--record FILE)."""

from __future__ import annotations

import json

from .telemetry.packet import Frame

SAMPLE_EVERY_S = 0.1  # 10 Hz keeps lap distance accurate enough; lap/flag changes are always written


class Recorder:
    def __init__(self, path: str):
        self._f = open(path, "w", buffering=1)  # line-buffered: every line reaches disk
        self._last_t = float("-inf")
        self._last_key: tuple | None = None
        self.written = 0

    def write(self, f: Frame) -> None:
        key = (f.lap, f.flags, f.total_laps, f.race_position, f.last_lap_ms, f.best_lap_ms)
        if key == self._last_key and f.t - self._last_t < SAMPLE_EVERY_S:
            return
        self._last_key, self._last_t = key, f.t
        self._f.write(
            json.dumps(
                {
                    "t": round(f.t, 3), "id": f.packet_id, "lap": f.lap, "laps": f.total_laps,
                    "flags": f.flags & 0xFFFF, "on_track": f.on_track, "pos": f.race_position, "cars": f.num_cars,
                    "fuel": round(f.fuel, 2), "kph": round(f.speed_kph, 1), "gear": f.gear,
                    "x": round(f.position[0], 2), "y": round(f.position[1], 2), "z": round(f.position[2], 2),
                    "ms": round(f.speed, 3),
                    "last_ms": f.last_lap_ms, "best_ms": f.best_lap_ms,
                }
            )
            + "\n"
        )
        self.written += 1

    def close(self) -> None:
        if not self._f.closed:
            self._f.close()
