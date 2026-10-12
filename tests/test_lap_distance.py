"""Lap length and progress. The real-race tests use a 10 Hz capture of a 10-lap, 20-car race (5.72 km lap)."""

import dataclasses
import gzip
import json
import math
from pathlib import Path

from gt7_engineer.state import NewLap, StateStore
from gt7_engineer.telemetry import simulator
from gt7_engineer.telemetry.packet import Frame

REAL = Path(__file__).parent / "data" / "ps5_full_race.jsonl.gz"
CIRCLE_M = 2 * math.pi * 100  # the simulator drives a 100 m radius circle


def sim(**kw):
    import asyncio

    async def go():
        return [f async for f in simulator.frames(realtime=False, **kw)]

    return asyncio.run(go())


def feed(frames):
    store, laps = StateStore(), []
    for f in frames:
        ev = store.update(f)
        if isinstance(ev, NewLap):
            laps.append(ev.record)
    return store, laps


def load_real() -> list[Frame]:
    out = []
    with gzip.open(REAL, "rt") as f:
        for line in f:
            r = json.loads(line)
            out.append(Frame(
                t=r["t"], packet_id=r["id"], position=(r["x"], r["y"], r["z"]), speed=r["ms"], rpm=0.0, fuel=r["fuel"],
                fuel_capacity=100.0, boost=0.0, oil_temp=110.0, water_temp=85.0, tyre_temp=(0.0,) * 4, lap=r["lap"],
                total_laps=r["laps"], best_lap_ms=r["best_ms"], last_lap_ms=r["last_ms"], race_position=r["pos"],
                num_cars=r["cars"], flags=r["flags"], gear=r["gear"], throttle=0.0, brake=0.0,
            ))
    return out


# ---------------------------------------------------------------- simulator
def test_lap_length_and_progress_in_the_simulator():
    frames = sim(lap_seconds=3, total_laps=5)
    store, laps = feed(frames)
    assert all(abs(l.distance_m - CIRCLE_M) < 0.03 * CIRCLE_M and l.distance_ok for l in laps)
    assert abs(store.lap_length_m - CIRCLE_M) < 0.03 * CIRCLE_M

    # halfway round the final lap the progress is about 50%
    store2 = StateStore()
    last = max(f.lap for f in frames)
    lap_frames = [f for f in frames if f.lap == last]
    for f in frames:
        store2.update(f)
        if f is lap_frames[len(lap_frames) // 2]:
            break
    assert 0.40 < store2.lap_progress < 0.60
    assert store2.summary()["lap_progress_pct"] in range(40, 61)


def test_progress_is_unknown_until_one_clean_lap_has_been_seen():
    frames = sim(lap_seconds=3, total_laps=4)
    store = StateStore()
    for f in frames[:120]:  # part of lap 1
        store.update(f)
    assert store.lap_length_m is None and store.lap_progress is None
    assert "lap_progress_pct" not in store.summary()


def test_joining_mid_lap_gives_progress_only_from_the_next_lap():
    frames = [f for f in sim(lap_seconds=3, total_laps=6) if f.lap >= 2]
    store = StateStore()
    for f in frames[:60]:  # joined partway round lap 2
        store.update(f)
    assert store.lap_progress is None  # we don't know where lap 2 began
    for f in frames[60:300]:  # across the first crossing and halfway round lap 3
        store.update(f)
    assert store.lap_length_m is None  # lap 2 was only partly seen; lap 3 is not finished yet
    assert store.lap_progress is None
    for f in frames[300:400]:  # lap 3 completes: now there is a clean lap to measure against
        store.update(f)
    assert store.lap_length_m is not None and store.lap_progress is not None


def test_dropped_packets_are_not_mistaken_for_a_teleport():
    frames = sim(lap_seconds=3, total_laps=3)
    # drop 15 packets mid-lap-2 (as the real UDP stream does): ids jump and the car moves 15 frames at once
    lossy = frames[:200] + frames[215:]
    store, laps = feed(lossy)
    assert all(l.distance_ok for l in laps)
    assert abs(laps[1].distance_m - CIRCLE_M) < 0.03 * CIRCLE_M


def test_a_teleport_marks_that_lap_unclean_and_the_median_ignores_it():
    frames = sim(lap_seconds=3, total_laps=6)
    # pit exit: car stationary, then placed 150 m away in one packet
    jump = []
    for f in frames:
        if f.lap == 3 and f.packet_id == 400:
            f = dataclasses.replace(f, speed=0.0, position=(f.position[0] + 150, f.position[1], f.position[2]))
        jump.append(f)
    store, laps = feed(jump)
    by_lap = {l.lap: l for l in laps}
    assert not by_lap[3].distance_ok and all(by_lap[n].distance_ok for n in (1, 2, 4, 5))
    assert abs(store.lap_length_m - CIRCLE_M) < 0.03 * CIRCLE_M


# ---------------------------------------------------------------- real race
def test_real_race_lap_length_matches_the_measured_5_72_km():
    store, laps = feed(load_real())
    assert len(laps) == 10
    clean = [l for l in laps if l.distance_ok]
    assert len(clean) >= 7 and all(5690 < l.distance_m < 5750 for l in clean)  # every clean lap within ~0.5%
    assert 5700 < store.lap_length_m < 5740
    assert not {6, 7} & {l.lap for l in clean}  # your pit in-lap and out-lap are flagged, not trusted


def test_real_race_progress_is_sane_through_a_lap():
    frames = load_real()
    lap9 = [f for f in frames if f.lap == 9]
    marks = {len(lap9) // 4: 0.25, len(lap9) // 2: 0.5, 3 * len(lap9) // 4: 0.75}  # frame index within lap 9
    store, seen, idx = StateStore(), [], 0
    for f in frames:
        store.update(f)
        if f.lap == 9:
            if idx in marks:
                seen.append((marks[idx], store.lap_progress))
            idx += 1
    assert len(seen) == 3 and all(p is not None for _, p in seen)
    for expected, got in seen:
        assert abs(got - expected) < 0.12, (expected, got)  # time-fraction vs distance-fraction: close, not equal
    assert [p for _, p in seen] == sorted(p for _, p in seen)  # and it only ever advances


def test_real_race_loses_no_distance_to_dropped_packets():
    store, laps = feed(load_real())
    lengths = [l.distance_m for l in laps if l.distance_ok]
    assert max(lengths) - min(lengths) < 50  # before the packet-id fix, laps with drops came out 100+ m short
