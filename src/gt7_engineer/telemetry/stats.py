"""Telemetry link health, shared between the UDP ingest, the app loop and the dashboard."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class IngestStats:
    received: int = 0  # raw UDP datagrams
    rejected: int = 0  # datagrams that failed decryption / magic check
    decoded: int = 0  # frames that made it into the state
    lost: int = 0  # packets GT7 sent that never reached us (gaps in its packet counter)
    _last_id: int | None = None
    last_frame: float | None = None  # monotonic time of the last decoded frame
    _times: deque = field(default_factory=lambda: deque(maxlen=240))

    def frame(self, packet_id: int | None = None) -> None:
        now = time.monotonic()
        if packet_id is not None:
            if self._last_id is not None and 1 < packet_id - self._last_id <= 600:
                self.lost += packet_id - self._last_id - 1
            self._last_id = packet_id
        self.decoded += 1
        self.last_frame = now
        self._times.append(now)

    def age(self, now: float | None = None) -> float | None:
        """Seconds since the last decoded frame; None if none yet."""
        if self.last_frame is None:
            return None
        return (now if now is not None else time.monotonic()) - self.last_frame

    def hz(self, now: float | None = None) -> float:
        """Decoded frames per second over the last second."""
        now = now if now is not None else time.monotonic()
        return float(sum(1 for t in self._times if now - t <= 1.0))
