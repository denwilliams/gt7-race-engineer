import asyncio
from pathlib import Path

import pytest

from gt7_engineer import app
from gt7_engineer.briefing import lap_update, pit_summary
from gt7_engineer.config import Config
from gt7_engineer.engineer import RaceEngineer
from gt7_engineer.pit import PitDetector, PitStop, TyreChange
from gt7_engineer.state import NewLap, StateStore
from gt7_engineer.telemetry import raw, simulator

REAL_PIT = str(Path(__file__).parent / "data" / "ps5_pit_stop.raw.jsonl.gz")


def sim(**kw):
    async def go():
        return [f async for f in simulator.frames(realtime=False, **kw)]

    return asyncio.run(go())


def detect(frames):
    det, out = PitDetector(), []
    for f in frames:
        out.extend(det.update(f))
    return out


# ---- each service is detected on its own: a stop can be fuel, tyres, both or neither
@pytest.mark.parametrize(
    "pit, added, tyres",
    [
        ({"lap": 3, "refuel": 30.0, "tyres": True}, 30.0, True),
        ({"lap": 3, "refuel": 30.0}, 30.0, False),  # fuel only
        ({"lap": 3, "tyres": True}, None, True),  # tyres only
    ],
)
def test_each_kind_of_stop_is_detected(pit, added, tyres):
    stops = [e for e in detect(sim(lap_seconds=20, total_laps=5, start_fuel=100.0, fuel_per_lap=7.0, pit=pit)) if isinstance(e, PitStop)]
    assert len(stops) == 1 and stops[0].lap == 3
    assert stops[0].tyres_changed is tyres
    assert (stops[0].added_l is None) if added is None else abs(stops[0].added_l - added) < 1.0


def test_a_stop_with_no_service_leaves_nothing_to_report_and_no_false_alarms():
    assert detect(sim(lap_seconds=20, total_laps=5, pit={"lap": 3})) == []  # sat still, nothing changed
    assert detect(sim(lap_seconds=20, total_laps=5, start_fuel=100.0)) == []  # ordinary racing


def test_fuel_jitter_while_driving_is_not_a_refuel():
    import dataclasses

    frames = sim(lap_seconds=10, total_laps=3, start_fuel=100.0)
    wobble = [dataclasses.replace(f, fuel=f.fuel + (0.4 if i % 7 == 0 else 0.0)) for i, f in enumerate(frames)]
    assert detect(wobble) == []


# ---- the real race
def test_real_pit_stop_is_found_exactly_once_with_the_real_numbers():
    async def go():
        return [f async for f in raw.frames(REAL_PIT)]

    events = detect(asyncio.run(go()))
    tyres = [e for e in events if isinstance(e, TyreChange)]
    stops = [e for e in events if isinstance(e, PitStop)]
    assert len(tyres) == 1 and len(stops) == 1
    assert abs(stops[0].added_l - 41.5) < 1.0 and stops[0].tyres_changed  # 21.7 L -> 63.2 L, new tyres
    assert 62 < stops[0].fuel_after < 64


# ---- what the store does with it
def feed(frames):
    store, laps = StateStore(), []
    for f in frames:
        ev = store.update(f)
        if isinstance(ev, NewLap):
            laps.append(ev)
    return store, laps


def test_store_clears_a_stale_wear_report_and_tracks_tyre_age():
    frames = sim(lap_seconds=20, total_laps=6, start_fuel=100.0, pit={"lap": 3, "refuel": 20.0, "tyres": True})
    store = StateStore()
    seen_age = {}
    for f in frames:
        store.update(f)
        if f.lap == 2 and "reported" not in seen_age:
            store.add_report("tyre_wear", "fronts about sixty percent")
            seen_age["reported"] = True
        if f.lap in (2, 4, 6):
            seen_age[f.lap] = store.tyre_age_laps
    assert store.latest_report("tyre_wear") is None  # wrong once the tyres were changed
    assert seen_age[2] == 1 and seen_age[4] == 0 and seen_age[6] == 2  # laps run on the current set
    sm = store.summary()
    assert sm["pit_stops"] == [{"lap": 3, "fuel_added_l": pytest.approx(20.0, abs=1.0), "new_tyres": True}]


def test_a_refuel_only_stop_keeps_the_wear_report_and_tyre_age():
    frames = sim(lap_seconds=20, total_laps=4, start_fuel=100.0, pit={"lap": 2, "refuel": 20.0})
    store = StateStore()
    for f in frames:
        store.update(f)
        if f.lap == 1 and f.packet_id == 400:
            store.add_report("tyre_wear", "rears forty")
    assert store.latest_report("tyre_wear") is not None  # same tyres: the report is still true
    assert store.summary()["pit_stops"][0]["new_tyres"] is False


# ---- what the engineer says
def test_pit_summary_only_mentions_what_happened():
    mk = lambda added, tyres: PitStop(3, 0.0, added, tyres, 50.0)
    assert pit_summary(mk(41.5, True)) == "Pit stop done. Added forty-two litres. New tyres."
    assert pit_summary(mk(20.0, False)) == "Pit stop done. Added twenty litres."
    assert pit_summary(mk(None, True)) == "Pit stop done. New tyres."


async def test_the_stop_is_said_once_inside_the_lap_update_not_twice():
    said = []

    async def speak(t):
        said.append(t)

    src = simulator.frames(realtime=False, lap_seconds=8, total_laps=4, start_fuel=100.0, pit={"lap": 2, "refuel": 20.0, "tyres": True, "stop_s": 5})
    await app.run(Config(), src, speak, None, use_llm=False)
    stops = [t for t in said if "Pit stop done" in t]
    assert len(stops) == 1  # either with the lap update or on its own, never both
    assert "New tyres." in stops[0] and "Added twenty litres." in stops[0]
