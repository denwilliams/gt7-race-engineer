"""UDP listener: sends GT7 heartbeats and yields decoded frames."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator

from .packet import Frame, parse

log = logging.getLogger(__name__)

SEND_PORT = 33739  # console listens here for the heartbeat
RECV_PORT = 33740  # console sends telemetry here
HEARTBEAT = b"A"
HEARTBEAT_EVERY = 1.5  # seconds; GT7 stops sending after ~10s of silence


class _Protocol(asyncio.DatagramProtocol):
    def __init__(self, queue: asyncio.Queue[bytes]):
        self.queue = queue

    def datagram_received(self, data: bytes, addr) -> None:
        if self.queue.full():
            self.queue.get_nowait()  # drop oldest; stay live
        self.queue.put_nowait(data)


async def frames(ps_ip: str, recv_port: int = RECV_PORT) -> AsyncIterator[Frame]:
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=600)
    transport, _ = await loop.create_datagram_endpoint(
        lambda: _Protocol(queue), local_addr=("0.0.0.0", recv_port)
    )

    async def heartbeat() -> None:
        while True:
            transport.sendto(HEARTBEAT, (ps_ip, SEND_PORT))
            await asyncio.sleep(HEARTBEAT_EVERY)

    hb = asyncio.create_task(heartbeat())
    try:
        while True:
            data = await queue.get()
            frame = parse(data, time.monotonic())
            if frame is None:
                log.debug("ignoring invalid packet (%d bytes)", len(data))
                continue
            yield frame
    finally:
        hb.cancel()
        transport.close()
