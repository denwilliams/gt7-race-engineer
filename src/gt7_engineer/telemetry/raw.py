"""Record and replay the decrypted GT7 packets themselves (every field, every packet).

One JSON line per packet: {"t": receive time, "p": base64 of the 296-byte decrypted packet}.
A path ending in .gz is gzip-compressed (about 5x smaller; a 25 minute race is ~7 MB).
"""

from __future__ import annotations

import asyncio
import base64
import gzip
import json
from collections.abc import AsyncIterator

from .packet import Frame, decode


FLUSH_EVERY = 120  # packets (~2 s at 60 Hz)


def _open(path: str, mode: str):
    return gzip.open(path, mode + "t") if path.endswith(".gz") else open(path, mode)


class RawRecorder:
    def __init__(self, path: str):
        self._f = _open(path, "w")
        self.written = 0

    def write(self, t: float, plain: bytes) -> None:
        self._f.write(json.dumps({"t": round(t, 4), "p": base64.b64encode(plain).decode()}) + "\n")
        self.written += 1
        if self.written % FLUSH_EVERY == 0:
            self._f.flush()  # for .gz this writes a sync point, so a crash loses ~2 s, not the whole tail

    def close(self) -> None:
        if not self._f.closed:
            self._f.close()


async def frames(path: str, realtime: bool = False, speedup: float = 1.0) -> AsyncIterator[Frame]:
    """Decode a raw recording back into Frames, optionally paced like the original race."""
    prev_t: float | None = None
    with _open(path, "r") as f:
        try:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue  # a half-written last line
                if realtime and prev_t is not None:
                    await asyncio.sleep(max(0.0, (r["t"] - prev_t) / speedup))
                prev_t = r["t"]
                yield decode(base64.b64decode(r["p"]), r["t"])
                if not realtime:
                    await asyncio.sleep(0)  # let other tasks run
        except EOFError:
            return  # gzip file that was never closed (recording killed): use what was readable
