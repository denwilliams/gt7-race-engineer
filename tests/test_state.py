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


def test_burn_trend_heavier_lighter_steady():
    heavier, _ = run_sim(lap_seconds=3, total_laps=7, burn_growth=0.10, start_fuel=200.0)
    assert heavier.burn_trend[1] == "heavier" and heavier.burn_trend[0] > 0
    lighter, _ = run_sim(lap_seconds=3, total_laps=7, burn_growth=-0.10, start_fuel=200.0)
    assert lighter.burn_trend[1] == "lighter" and lighter.burn_trend[0] < 0
    steady, _ = run_sim(lap_seconds=3, total_laps=7, start_fuel=200.0)
    assert steady.burn_trend[1] == "steady"
    assert "heavier" in heavier.summary()["fuel_burn_trend"]
    assert len(heavier.burns) == 5 and heavier.last_lap_burn == heavier.burns[-1]


def test_trend_needs_three_laps():
    store, _ = run_sim(lap_seconds=3, total_laps=3)  # 2 completed laps
    assert store.burn_trend is None


def test_pit_stop_lap_excluded_from_burn():
    store, _ = run_sim(lap_seconds=3, total_laps=6, start_fuel=40.0, fuel_per_lap=7.0, refuel_lap=3)
    assert [l.burn_valid for l in store.laps] == [True, True, False, True, True]
    assert abs(store.fuel_per_lap - 7.0) < 0.3
    assert all(abs(b - 7.0) < 0.3 for b in store.burns)


def test_summary_has_fuel_margin():
    store, _ = run_sim(lap_seconds=3, total_laps=6, start_fuel=40.0, fuel_per_lap=7.0)
    sm = store.summary()
    assert sm["laps_remaining_including_current"] == 1  # final lap in progress
    assert "fuel_margin_laps" in sm


def test_fuel_at_start_of_each_lap():
    store, events = run_sim(lap_seconds=3, total_laps=5, start_fuel=40.0, fuel_per_lap=7.0)
    starts = store.fuel_at_lap_start
    assert sorted(starts) == [1, 2, 3, 4, 5]  # every lap seen, including the one in progress
    for lap in range(1, 6):
        assert abs(starts[lap] - (40.0 - 7.0 * (lap - 1))) < 0.4
    assert store.fuel_at_start_of(3) == starts[3] and store.fuel_at_start_of(9) is None
    assert [round(e.record.fuel_start) for e in events] == [40, 33, 26, 19]  # on each completed lap too
    assert store.summary()["fuel_at_lap_start_l"][2] == round(starts[2], 1)


def test_fuel_at_lap_start_after_pit_stop_shows_the_refuel():
    store, _ = run_sim(lap_seconds=3, total_laps=6, start_fuel=40.0, fuel_per_lap=7.0, refuel_lap=3)
    assert store.fuel_at_start_of(4) > store.fuel_at_start_of(3)  # tank is fuller after the stop


def test_start_position_is_remembered_after_gt7_stops_reporting_it():
    store, _ = run_sim(lap_seconds=3, total_laps=3)
    assert store.car.start_position == -1  # live frames no longer carry it, as in real GT7
    sm = store.summary()
    assert sm["starting_position"] == 3 and sm["cars_in_race"] == 12
    assert "not in telemetry" in sm["current_position"] or "unknown" in sm["current_position"]


def test_driver_reported_position_reaches_summary():
    store, _ = run_sim(lap_seconds=3, total_laps=3)
    store.add_report("position", "P2 at lap 2")
    assert "P2 at lap 2" in store.summary()["current_position"]
