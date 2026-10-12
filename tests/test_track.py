"""Turn detection, the turn map and lap comparison. Synthetic tracks (known answers) and a real race."""

import gzip
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from gt7_engineer.state import NewLap, StateStore, TrackMapped
from gt7_engineer.telemetry.packet import Frame
from gt7_engineer.track import (
    MapTurn, TrackModel, TurnPass, compare_passes, consensus, detect_turns, make_profile,
)

REAL = Path(__file__).parent / "data" / "ps5_clean_laps.jsonl.gz"


# ------------------------------------------------------------------ a synthetic track with known turns
# (turn-in distance, heading change in degrees (+ = the "positive" side), radius in metres)
TURNS = [(500, +90, 80), (1000, -90, 80), (1500, +180, 60)]
LENGTH = 2200
BASE_V = 60.0


def synth_lap(lap, *, vmin=None, gears=None, brake_m=None, thr_m=None, wiggle=False, t0=0.0, pid0=0):
    """Drive TURNS once. Per-turn dicts give the slowest speed (m/s), gear, braking distance and throttle pickup."""
    vmin = vmin or {}
    gears = gears or {}
    brake_m = brake_m or {}
    thr_m = thr_m or {}
    x = z = heading = 0.0
    t, frames = t0, []
    arcs = [(s0, s0 + abs(math.radians(a)) * r, math.copysign(1 / r, a)) for s0, a, r in TURNS]
    if wiggle:  # a gentle 10 degree wobble that must NOT count as a turn
        arcs.append((1300, 1300 + 40, math.radians(10) / 40))
    for s in range(LENGTH):
        k = next((kk for a, b, kk in arcs if a <= s < b), 0.0)
        heading += k
        x, z = x + math.cos(heading), z + math.sin(heading)
        v, gear, brake, throttle = BASE_V, 5, 0.0, 1.0
        for n, (s0, a, r) in enumerate(TURNS, 1):
            apex = s0 + abs(math.radians(a)) * r / 2
            end = s0 + abs(math.radians(a)) * r
            vm = vmin.get(n, 30.0)
            dist = abs(s - apex)
            if dist < 120:
                v = min(v, vm + (BASE_V - vm) * dist / 120)
            if s0 - 150 < s < end + 80:
                gear = gears.get(n, 3)
            bm = brake_m.get(n, 100.0)
            if s0 - bm <= s < apex:
                brake, throttle = 0.8, 0.0
            elif apex <= s < apex + thr_m.get(n, 30.0):
                throttle = 0.3
        t += 1.0 / v
        frames.append(Frame(
            t=t, packet_id=pid0 + s, position=(x, 10.0, z), speed=v, rpm=0, fuel=50, fuel_capacity=100, boost=0,
            oil_temp=0, water_temp=0, tyre_temp=(60,) * 4, lap=lap, total_laps=10, best_lap_ms=0, last_lap_ms=0,
            race_position=1, num_cars=1, flags=0x18, gear=gear, throttle=throttle, brake=brake,
        ))
    return frames


def test_detects_the_turns_directions_and_places_on_a_known_track():
    turns = detect_turns(make_profile(synth_lap(1)))
    assert [t.sign for t in turns] == [+1, -1, +1]
    for t, (s0, a, r) in zip(turns, TURNS):
        apex = s0 + abs(math.radians(a)) * r / 2
        assert abs(t.apex_s - apex) < 40, (t.apex_s, apex)
        assert abs(abs(t.angle_deg) - abs(a)) < 15, (t.angle_deg, a)


def test_a_small_wobble_is_not_a_turn():
    assert len(detect_turns(make_profile(synth_lap(1, wiggle=True)))) == 3


def test_the_map_needs_most_laps_to_agree_and_ignores_a_one_off():
    tm = TrackModel()
    for lap in range(1, 6):
        tm.add_lap(lap, synth_lap(lap, wiggle=(lap == 3)))  # one lap has a wobble: still 3 turns
    assert [t.number for t in tm.turns] == [1, 2, 3] and all(t.seen_on == 5 for t in tm.turns)
    assert [t.label() for t in tm.turns] == ["L", "R", "L"]
    assert [t.label(mirror=True) for t in tm.turns] == ["R", "L", "R"]  # the mirror switch flips them


def test_next_turn_ahead_and_wraps_round_the_lap():
    tm = TrackModel()
    for lap in range(1, 4):
        tm.add_lap(lap, synth_lap(lap))
    turn, dist = tm.next_turn(100)
    assert turn.number == 1 and 300 < dist < 450  # ~400 m to turn 1
    turn, dist = tm.next_turn(1900)  # past the last turn: wraps to turn 1 of the next lap
    assert turn.number == 1 and dist > 500


# ------------------------------------------------------------------ comparing laps
def build(variants):
    tm = TrackModel()
    for lap, kw in enumerate(variants, 1):
        tm.add_lap(lap, synth_lap(lap, **kw))
    return tm


def test_finds_that_a_higher_gear_and_a_later_brake_point_were_faster():
    slow = dict(vmin={2: 24.0}, gears={2: 2}, brake_m={2: 140.0})
    fast = dict(vmin={2: 33.0}, gears={2: 3}, brake_m={2: 100.0})
    tm = build([slow, fast, slow, fast, fast, slow, fast, slow])
    ins = {i.turn: i for i in tm.insights()}
    assert 2 in ins and "gear 3 instead of 2" in ins[2].text and "later" in ins[2].text and ins[2].gap_s > 0.15
    assert 1 not in ins and 3 not in ins  # nothing differed there


def test_an_outlier_lap_does_not_become_the_slow_pattern():
    """One lap where you run wide (very slow, odd gear) must not be blamed on the gear the other laps also used."""
    normal = dict(vmin={2: 30.0}, gears={2: 3})
    spin = dict(vmin={2: 8.0}, gears={2: 1}, brake_m={2: 160.0})
    tm = build([normal] * 7 + [spin])
    assert not any(i.turn == 2 for i in tm.insights())


def test_too_few_laps_gives_no_conclusions():
    tm = build([dict(vmin={2: 24.0}, gears={2: 2}), dict(vmin={2: 33.0}, gears={2: 3})])
    assert tm.insights() == []


# ------------------------------------------------------------------ the real race
def load_real():
    laps = defaultdict(list)
    for line in gzip.open(REAL, "rt"):
        r = json.loads(line)
        laps[r["l"]].append(Frame(
            t=r["t"], packet_id=r["i"], position=(r["x"], 12.0, r["z"]), speed=r["v"], rpm=0, fuel=50, fuel_capacity=100,
            boost=0, oil_temp=110, water_temp=85, tyre_temp=(60,) * 4, lap=r["l"], total_laps=10, best_lap_ms=0, last_lap_ms=0,
            race_position=1, num_cars=20, flags=0x18, gear=r["g"], throttle=r["th"], brake=r["br"],
        ))
    return laps


# turn apexes (m from the line) as found on every lap of the real race, and the side of each
REAL_APEX = [302, 410, 655, 850, 1585, 2048, 2185, 2312, 3538, 3895, 4108, 4625, 4880, 5260]
REAL_SIDES = ["R", "L", "R", "L", "L", "R", "L", "R", "L", "R", "L", "L", "R", "R"]


def test_real_race_map_has_the_same_14_turns_wherever_you_look():
    laps = load_real()
    tm = TrackModel()
    for lap in sorted(laps):
        tm.add_lap(lap, laps[lap])
    assert len(tm.turns) == 14
    for t, apex, side in zip(tm.turns, REAL_APEX, REAL_SIDES):
        assert abs(t.apex_s - apex) < 30 and t.label() == side
        assert t.seen_on >= 7  # seen on at least 7 of the 8 laps


def test_real_race_map_keeps_all_14_real_turns_from_the_fourth_lap():
    """A marginal ~25 degree kink may be in or out of the map until more laps settle it; the 14 real turns never are."""
    laps = load_real()
    tm = TrackModel()
    for n, lap in enumerate(sorted(laps), 1):
        tm.add_lap(lap, laps[lap])
        if n >= 4:
            apexes = [t.apex_s for t in tm.turns]
            for expected in REAL_APEX:
                assert any(abs(a - expected) < 30 for a in apexes), (n, expected, apexes)
            assert 14 <= len(tm.turns) <= 15
    assert len(tm.turns) == 14  # and with eight laps the marginal one is gone


def test_real_race_finds_the_gear_and_braking_difference_at_turn_5():
    laps = load_real()
    tm = TrackModel()
    for lap in sorted(laps):
        tm.add_lap(lap, laps[lap])
    t5 = next(i for i in tm.insights() if i.turn == 5)
    assert "gear 4 instead of 2" in t5.text and "earlier" in t5.text and t5.gap_s > 0.4


# ------------------------------------------------------------------ in the live state
def test_the_state_learns_the_turns_as_clean_laps_complete():
    laps = load_real()
    store = StateStore()
    mapped = []
    for lap in (1, 2, 3, 4, 5):
        for f in laps[lap]:
            store.update(f)
            while store.events:
                ev = store.events.popleft()
                if isinstance(ev, TrackMapped):
                    mapped.append(ev)
    assert mapped and mapped[-1].turns >= 13  # four completed laps: the map is there
    sm = store.summary()
    assert len(sm["track_turns"]) >= 13 and sm["track_turns"][0] == {"n": 1, "dir": "R", "apex_m": pytest_approx(302), "angle": pytest_approx(84)}
    assert sm["next_turn"]["number"] >= 1 and sm["next_turn"]["in_m"] >= 0


def pytest_approx(v):
    import pytest

    return pytest.approx(v, abs=30)
