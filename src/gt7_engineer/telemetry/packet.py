"""GT7 UDP telemetry packet decryption and decoding.

Format follows the community reverse-engineering of the "Simulator Interface"
packet (type A, 296 bytes). Offsets have not been verified against real
hardware by this project; encode_packet() exists so the decoder can be
round-trip tested and so the simulator can emit realistic packets.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from Crypto.Cipher import Salsa20

MAGIC = 0x47375330  # "0S7G" little-endian
PACKET_SIZE = 296
KEY = b"Simulator Interface Packet GT7 ver 0.0"[:32]
IV_XOR = 0xDEADBEAF
IV_OFFSET = 0x40

FLAG_ON_TRACK = 1 << 0
FLAG_PAUSED = 1 << 1
FLAG_LOADING = 1 << 2


@dataclass(slots=True)
class Frame:
    """One decoded telemetry packet. Units: m/s, litres, degrees C, ms."""

    t: float  # receive time (monotonic seconds); set by the ingest layer
    packet_id: int
    position: tuple[float, float, float]
    speed: float  # m/s
    rpm: float
    fuel: float
    fuel_capacity: float
    boost: float
    oil_temp: float
    water_temp: float
    tyre_temp: tuple[float, float, float, float]  # FL FR RL RR
    lap: int  # current lap, 1-based; 0 before the race starts
    total_laps: int  # 0 for timed races
    best_lap_ms: int  # -1 if none
    last_lap_ms: int  # -1 if none
    race_position: int  # 0x84. Documented as the pre-race start position, but real GT7 shows the LIVE position
    num_cars: int  # 0x86. Cars in the race; -1 when the mode doesn't report it
    flags: int
    gear: int
    throttle: float  # 0..1
    brake: float  # 0..1
    car_code: int = 0  # GT7's id for the car model (0x124); the same in every packet of a session

    @property
    def on_track(self) -> bool:
        """True while the session is running (not paused, not loading).

        GT7's own "car on track" bit (bit 0) is NOT reliably set during real races (observed
        flags 0x0418 while driving a race), so it is not used. See `car_on_track_bit`.
        """
        return not self.flags & (FLAG_PAUSED | FLAG_LOADING)

    @property
    def car_on_track_bit(self) -> bool:
        return bool(self.flags & FLAG_ON_TRACK)

    @property
    def speed_kph(self) -> float:
        return self.speed * 3.6


def _salsa(iv1: int, data: bytes) -> bytes:
    iv2 = iv1 ^ IV_XOR
    nonce = struct.pack("<II", iv2, iv1)
    return Salsa20.new(key=KEY, nonce=nonce).decrypt(data)


def decrypt(data: bytes) -> bytes | None:
    """Decrypt a raw packet; returns None if it is not a valid GT7 packet."""
    if len(data) < PACKET_SIZE:
        return None
    iv1 = struct.unpack_from("<I", data, IV_OFFSET)[0]
    plain = _salsa(iv1, data[:PACKET_SIZE])
    if struct.unpack_from("<I", plain, 0)[0] != MAGIC:
        return None
    return plain


def decode(plain: bytes, t: float = 0.0) -> Frame:
    u = struct.unpack_from
    return Frame(
        t=t,
        packet_id=u("<i", plain, 0x70)[0],
        position=u("<3f", plain, 0x04),
        speed=u("<f", plain, 0x4C)[0],
        rpm=u("<f", plain, 0x3C)[0],
        fuel=u("<f", plain, 0x44)[0],
        fuel_capacity=u("<f", plain, 0x48)[0],
        boost=u("<f", plain, 0x50)[0] - 1.0,
        oil_temp=u("<f", plain, 0x5C)[0],
        water_temp=u("<f", plain, 0x58)[0],
        tyre_temp=u("<4f", plain, 0x60),
        lap=u("<h", plain, 0x74)[0],
        total_laps=u("<h", plain, 0x76)[0],
        best_lap_ms=u("<i", plain, 0x78)[0],
        last_lap_ms=u("<i", plain, 0x7C)[0],
        race_position=u("<h", plain, 0x84)[0],
        num_cars=u("<h", plain, 0x86)[0],
        flags=u("<h", plain, 0x8E)[0],
        gear=u("<B", plain, 0x90)[0] & 0x0F,
        throttle=u("<B", plain, 0x91)[0] / 255.0,
        brake=u("<B", plain, 0x92)[0] / 255.0,
        car_code=u("<i", plain, 0x124)[0],
    )


def parse(data: bytes, t: float = 0.0) -> Frame | None:
    plain = decrypt(data)
    return decode(plain, t) if plain else None


def encode_packet(f: Frame, iv1: int = 0x1234ABCD) -> bytes:
    """Build an encrypted packet from a Frame (used by the simulator/tests)."""
    b = bytearray(PACKET_SIZE)
    p = struct.pack_into
    p("<I", b, 0, MAGIC)
    p("<3f", b, 0x04, *f.position)
    p("<f", b, 0x3C, f.rpm)
    p("<f", b, 0x44, f.fuel)
    p("<f", b, 0x48, f.fuel_capacity)
    p("<f", b, 0x4C, f.speed)
    p("<f", b, 0x50, f.boost + 1.0)
    p("<f", b, 0x58, f.water_temp)
    p("<f", b, 0x5C, f.oil_temp)
    p("<4f", b, 0x60, *f.tyre_temp)
    p("<i", b, 0x70, f.packet_id)
    p("<h", b, 0x74, f.lap)
    p("<h", b, 0x76, f.total_laps)
    p("<i", b, 0x78, f.best_lap_ms)
    p("<i", b, 0x7C, f.last_lap_ms)
    p("<h", b, 0x84, f.race_position)
    p("<h", b, 0x86, f.num_cars)
    p("<h", b, 0x8E, f.flags)
    p("<B", b, 0x90, f.gear & 0x0F)
    p("<B", b, 0x91, round(f.throttle * 255))
    p("<B", b, 0x92, round(f.brake * 255))
    p("<i", b, 0x124, f.car_code)
    # Salsa20 is symmetric: "decrypting" the plaintext encrypts it. The IV
    # travels in the clear at IV_OFFSET, so it is written after encryption.
    out = bytearray(_salsa(iv1, bytes(b)))
    struct.pack_into("<I", out, IV_OFFSET, iv1)
    return bytes(out)
