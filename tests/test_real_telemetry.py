"""Regression tests built from a real capture of a GT7 race on a PS5 (lap 3 -> 4 crossing, 20 cars)."""

import json
from pathlib import Path

from gt7_engineer.briefing import lap_update
from gt7_engineer.state import NewLap, StateStore
from gt7_engineer.telemetry.packet import Frame

DATA = Path(__file__).parent / "data" / "ps5_lap_crossing.jsonl"


def load() -> list[Frame]:
    out = []
    for line in DATA.read_text().splitlines():
        r = json.loads(line)
        out.append(Frame(
            t=r["t"], packet_id=r["id"], position=(0.0, 0.0, 0.0), speed=r["kph"] / 3.6, rpm=0.0, fuel=r["fuel"],
            fuel_capacity=100.0, boost=0.0, oil_temp=110.0, water_temp=85.0, tyre_temp=(0.0, 0.0, 0.0, 0.0),
            lap=r["lap"], total_laps=r["laps"], best_lap_ms=r["best_ms"], last_lap_ms=r["last_ms"],
            race_position=r["pos"], num_cars=r["cars"], flags=r["flags"], gear=r["gear"], throttle=0.0, brake=0.0,
        ))
    return out


def replay():
    store, events = StateStore(), []
    for f in load():
        ev = store.update(f)
        if ev is not None:
            events.append((ev, store.summary(), lap_update(store, ev) if isinstance(ev, NewLap) else None))
    return store, events


def test_real_flags_never_have_the_on_track_bit_but_frames_count_as_running():
    frames = load()
    assert not any(f.car_on_track_bit for f in frames)  # GT7 never set bit 0 in this real race
    assert all(f.on_track for f in frames)  # so running must not depend on it
    assert {f.flags for f in frames} <= {0x10, 0x18, 0x38, 0x410, 0x418, 0x430, 0x438}


def test_real_lap_crossing_is_detected_with_the_right_lap_time():
    store, events = replay()
    laps = [e for e, _, _ in events if isinstance(e, NewLap)]
    assert len(laps) == 1
    rec = laps[0].record
    assert rec.lap == 3 and rec.time_ms == 113836  # last_lap_ms updates in the same packet as the lap counter
    assert not rec.burn_valid  # the capture joined lap 3 part-way through
    assert not [e for e, _, _ in events if not isinstance(e, NewLap)]  # no false new-race reset


def test_real_position_is_live_and_tracked_per_lap():
    store, _ = replay()
    assert store.num_cars == 20
    assert store.position == 2  # moved 7 -> 3 -> 2 during the capture
    assert store.position_at_lap_start == {4: 3}  # lap 3's start was before we joined, so not claimed
    assert 3 not in store.fuel_at_lap_start and abs(store.fuel_at_lap_start[4] - 60.6) < 0.2


def test_fuel_estimate_is_available_right_after_joining_mid_race():
    store, events = replay()
    assert store.burns == []  # no complete lap yet
    est = store.estimated_fuel_per_lap
    assert est is not None and 12.5 < est < 14.5  # real burn: 17.65 L net over 150 s at a 1:54 lap = ~13.4 L/lap
    sm = store.summary()
    assert sm["fuel_per_lap_is_estimate"] is True and sm["laps_of_fuel"] is not None
    _, _, update = [e for e in events if isinstance(e[0], NewLap)][0]
    assert update.startswith("Lap four of ten.") and "Fuel about" in update and "Position three" in update
    assert "Last lap one fifty-three point eight" in update


def test_real_fuel_jitter_is_not_counted_as_burn():
    """The real fuel signal wobbles upward by up to ~0.4 L; summing every drop overcounted by 29%."""
    store, _ = replay()
    frames = load()
    net = frames[0].fuel - frames[-1].fuel  # 17.65 L actually left the tank
    burned = store._fuel_burned  # litres watched leaving the tank (the summary adds the assumed pre-join part)
    assert abs(burned - net) < 1.0, (burned, net)  # within ~5%, not 22.8
    assert not store._refuelled  # and none of the wobble looked like a pit stop


def test_a_real_refuel_still_resets_the_baseline_and_is_not_subtracted():
    import dataclasses

    frames = load()
    refuelled = frames[:300] + [dataclasses.replace(f, fuel=f.fuel + 40.0) for f in frames[300:600]]  # +40 L pit stop
    store = StateStore()
    for f in refuelled:
        store.update(f)
    expected = frames[0].fuel - frames[299].fuel + (frames[300].fuel - frames[599].fuel)  # drops either side of the stop
    assert abs(store._fuel_burned - expected) < 1.0


def test_joining_mid_race_assumes_a_full_tank_at_the_start_and_says_so():
    store, _ = replay()
    sm = store.summary()
    assert sm["fuel_at_race_start_l"] == 100.0  # the tank capacity in this capture: races start full
    assert "assumed a full tank" in sm["fuel_at_race_start_note"] and "lap 3" in sm["fuel_at_race_start_note"]
    # burned so far = pre-join (100 - 72.55) + watched (17.6) = whole race to date, i.e. 100 - now
    assert abs(sm["fuel_burned_total_l"] - (100.0 - sm["fuel_l"])) < 1.0
