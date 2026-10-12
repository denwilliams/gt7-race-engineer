import asyncio

from gt7_engineer.state import StateStore
from gt7_engineer.telemetry import simulator


def run_sim(**kw):
    store, events = StateStore(), []

    async def go():
        async for f in simulator.frames(realtime=False, **kw):
            ev = store.update(f)
            if ev:
                events.append(ev)

    asyncio.run(go())
    return store, events


def test_laps_detected_and_fuel_tracked():
    store, events = run_sim(lap_seconds=5, total_laps=4, fuel_per_lap=7.0)
    assert [e.record.lap for e in events] == [1, 2, 3]
    assert all(abs(e.record.fuel_used - 7.0) < 0.3 for e in events)
    assert abs(store.fuel_per_lap - 7.0) < 0.3
    assert store.laps_remaining == 1
    assert len(store.laps[0].frames) > 100


def test_recent_looks_back_in_time():
    store, _ = run_sim(lap_seconds=5, total_laps=2)
    last_second = store.recent(1.0)
    assert 50 <= len(last_second) <= 62
    assert store.recent(1000)[0].t < last_second[0].t


def test_driver_report_in_summary():
    store, _ = run_sim(lap_seconds=2, total_laps=2)
    assert "driver_reported_tyre_wear" not in store.summary()
    store.add_report("tyre_wear", "fronts about sixty percent")
    assert "sixty" in store.summary()["driver_reported_tyre_wear"]
