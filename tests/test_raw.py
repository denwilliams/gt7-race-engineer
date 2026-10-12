import asyncio

import pytest

from gt7_engineer import app
from gt7_engineer.config import Config
from gt7_engineer.state import StateStore
from gt7_engineer.telemetry import packet, raw, simulator


async def sim(**kw):
    return [f async for f in simulator.frames(realtime=False, **kw)]


def to_plain(f):
    return packet.decrypt(packet.encode_packet(f))  # what the ingest layer hands to the raw recorder


@pytest.mark.parametrize("name", ["race.jsonl", "race.jsonl.gz"])
async def test_raw_roundtrip_keeps_every_decoded_field(tmp_path, name):
    path = str(tmp_path / name)
    original = await sim(lap_seconds=2, total_laps=3)
    rec = raw.RawRecorder(path)
    for f in original:
        rec.write(f.t, to_plain(f))
    rec.close()
    assert rec.written == len(original)
    back = [f async for f in raw.frames(path)]
    assert len(back) == len(original)
    for a, b in zip(original[::37], back[::37]):
        assert (b.lap, b.total_laps, b.race_position, b.num_cars, b.flags, b.gear) == (a.lap, a.total_laps, a.race_position, a.num_cars, a.flags, a.gear)
        assert abs(b.fuel - a.fuel) < 1e-3 and abs(b.speed - a.speed) < 1e-3 and abs(b.t - a.t) < 1e-3
        assert all(abs(x - y) < 1e-3 for x, y in zip(a.position, b.position))


async def test_replayed_race_gives_the_same_laps_as_the_original(tmp_path):
    path = str(tmp_path / "race.jsonl.gz")
    original = await sim(lap_seconds=3, total_laps=4, start_fuel=40.0)
    rec = raw.RawRecorder(path)
    for f in original:
        rec.write(f.t, to_plain(f))
    rec.close()

    direct, replayed = StateStore(), StateStore()
    for f in original:
        direct.update(f)
    async for f in raw.frames(path):
        replayed.update(f)
    assert [l.lap for l in replayed.laps] == [l.lap for l in direct.laps] == [1, 2, 3]
    assert [round(l.fuel_used, 2) for l in replayed.laps] == [round(l.fuel_used, 2) for l in direct.laps]

    said = []

    async def speak(t):
        said.append(t)

    store = await app.run(Config(), raw.frames(path), speak, None, use_llm=False)  # replay through the whole app
    assert len(store.laps) == 3 and any(t.startswith("Lap two of four.") for t in said)


async def test_replay_realtime_paces_like_the_original(tmp_path):
    import time

    path = str(tmp_path / "r.jsonl")
    original = await sim(lap_seconds=1, total_laps=1)
    rec = raw.RawRecorder(path)
    for f in original[:30]:  # 0.5 s of race
        rec.write(f.t, to_plain(f))
    rec.close()
    t0 = time.time()
    n = len([f async for f in raw.frames(path, realtime=True)])
    assert n == 30 and 0.35 < time.time() - t0 < 1.5
    t0 = time.time()
    n = len([f async for f in raw.frames(path, realtime=True, speedup=10)])
    assert time.time() - t0 < 0.3


async def test_truncated_gzip_recording_is_still_readable(tmp_path):
    """A recording killed mid-race (gzip never closed) must still replay what was written."""
    path = str(tmp_path / "killed.jsonl.gz")
    original = await sim(lap_seconds=2, total_laps=3)
    rec = raw.RawRecorder(path)
    for f in original:
        rec.write(f.t, to_plain(f))  # never closed, like a SIGKILL
    back = [f async for f in raw.frames(path)]
    assert len(back) >= raw.FLUSH_EVERY and len(back) <= len(original)  # everything up to the last flush
    rec.close()


def test_ctrl_c_closes_recordings_so_the_gzip_is_complete(tmp_path):
    import os
    import signal
    import subprocess
    import sys
    import time

    out = tmp_path / "race.raw.jsonl.gz"
    rec = tmp_path / "race.jsonl"
    p = subprocess.Popen(
        [sys.executable, "-m", "gt7_engineer.cli", "run", "--simulate", "--no-audio", "--no-llm", "--plain", "--laps", "30",
         "--lap-seconds", "2", "--record-raw", str(out), "--record", str(rec)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(4)
    p.send_signal(signal.SIGINT)
    assert p.wait(timeout=10) == 0
    import gzip

    with gzip.open(out, "rt") as f:  # a properly finished gzip: reading to the end must not raise
        lines = f.read().splitlines()
    assert len(lines) > 100 and rec.stat().st_size > 0


def test_replay_with_no_audio_never_opens_the_microphone(tmp_path):
    import subprocess
    import sys

    path = str(tmp_path / "r.jsonl.gz")
    subprocess.run(
        [sys.executable, "-m", "gt7_engineer.cli", "run", "--simulate", "--fast", "--no-audio", "--no-llm", "--plain",
         "--laps", "3", "--lap-seconds", "3", "--record-raw", path],
        check=True, capture_output=True, timeout=60,
    )
    out = subprocess.run(
        [sys.executable, "-m", "gt7_engineer.cli", "run", "--replay", path, "--fast", "--no-audio", "--no-llm", "--plain"],
        capture_output=True, text=True, timeout=60,
    ).stdout
    assert "Lap two of three." in out and "🏁 Lap 1" in out
    assert "Listening" not in out and "Loading speech" not in out  # no mic, no models
