from gt7_engineer.telemetry import packet, simulator
import asyncio


async def first_frame():
    async for f in simulator.frames(realtime=False):
        return f


def test_round_trip():
    f = asyncio.run(first_frame())
    raw = packet.encode_packet(f)
    assert len(raw) == packet.PACKET_SIZE
    assert raw != packet.encode_packet(f, iv1=1)  # IV changes ciphertext
    g = packet.parse(raw, t=1.5)
    assert g is not None and g.t == 1.5
    assert g.lap == f.lap and g.total_laps == f.total_laps
    assert abs(g.fuel - f.fuel) < 1e-4 and abs(g.speed - f.speed) < 1e-4
    assert g.on_track and g.gear == f.gear
    assert all(abs(a - b) < 1e-4 for a, b in zip(g.tyre_temp, f.tyre_temp))


def test_rejects_garbage():
    assert packet.parse(b"\x00" * 10) is None
    assert packet.parse(bytes(range(256)) + bytes(40)) is None
