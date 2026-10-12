"""UDP listener: sends GT7 heartbeats and yields decoded frames."""

from __future__ import annotations

import asyncio
import logging
import socket
import time
from collections.abc import AsyncIterator

from .packet import Frame, decode, decrypt
from .stats import IngestStats

log = logging.getLogger(__name__)

SEND_PORT = 33739  # console listens here for the heartbeat
RECV_PORT = 33740  # console sends telemetry here
RCVBUF_BYTES = 4 * 1024 * 1024
HEARTBEAT = b"A"
HEARTBEAT_EVERY = 1.5  # seconds; GT7 stops sending after ~10s of silence


class _Protocol(asyncio.DatagramProtocol):
    def __init__(self, queue: asyncio.Queue[bytes]):
        self.queue = queue

    def datagram_received(self, data: bytes, addr) -> None:
        if self.queue.full():
            self.queue.get_nowait()  # drop oldest; stay live
        self.queue.put_nowait(data)


async def frames(
    ps_ip: str, recv_port: int = RECV_PORT, stats: IngestStats | None = None, raw_sink=None
) -> AsyncIterator[Frame]:
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=600)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # The default buffer holds ~30 packets (half a second); a 0.2 s stall in this process then
    # silently drops a dozen. Ask for 4 MB (minutes of telemetry).
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, RCVBUF_BYTES)
    sock.bind(("0.0.0.0", recv_port))
    sock.setblocking(False)
    transport, _ = await loop.create_datagram_endpoint(lambda: _Protocol(queue), sock=sock)

    async def heartbeat() -> None:
        while True:
            transport.sendto(HEARTBEAT, (ps_ip, SEND_PORT))
            await asyncio.sleep(HEARTBEAT_EVERY)

    hb = asyncio.create_task(heartbeat())
    try:
        while True:
            data = await queue.get()
            if stats is not None:
                stats.received += 1
            now = time.monotonic()
            plain = decrypt(data)
            frame = decode(plain, now) if plain else None
            if plain and raw_sink is not None:
                raw_sink(now, plain)  # every field of every valid packet, for offline analysis
            if frame is None:
                if stats is not None:
                    stats.rejected += 1
                log.debug("ignoring invalid packet (%d bytes)", len(data))
                continue
            yield frame
    finally:
        hb.cancel()
        transport.close()
