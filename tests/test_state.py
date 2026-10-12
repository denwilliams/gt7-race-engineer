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


def test_live_position_is_tracked_with_start_and_per_lap_history():
    store, _ = run_sim(lap_seconds=3, total_laps=5, grid_position=8)
    assert store.position == 4 and store.num_cars == 12  # 8 on the grid, +1 place per lap, lap 5 now
    assert store.start_position == 8
    assert store.position_at_lap_start == {1: 8, 2: 7, 3: 6, 4: 5, 5: 4}
    sm = store.summary()
    assert sm["position"] == 4 and sm["cars_in_race"] == 12 and sm["starting_position"] == 8
    assert "position_note" not in sm  # telemetry has it, no need to ask the driver


def test_prerace_only_mode_falls_back_to_the_driver():
    store, _ = run_sim(lap_seconds=3, total_laps=3, position_mode="prerace")
    assert store.car.race_position == -1  # a mode that stops reporting position
    assert store.start_position == 8  # still remembered from the grid
    assert "ask the driver" in store.summary()["position_note"]
    store.add_report("position", "P2 at lap 2")
    assert "P2 at lap 2" in store.summary()["position_note"]




def test_summary_has_formatted_lap_times_and_fuel_burned():
    store, _ = run_sim(lap_seconds=3, total_laps=5, start_fuel=40.0, fuel_per_lap=7.0)
    sm = store.summary()
    assert "last_lap_ms" not in sm and "best_lap_ms" not in sm  # raw ms invite conversion mistakes
    assert sm["last_lap_time"].startswith("0:0") and sm["best_lap_time"].startswith("0:0")
    assert isinstance(sm["last_lap_vs_best_s"], float)
    assert sm["fuel_at_race_start_l"] == 40.0
    # burned so far = start - now, whatever the lap
    assert abs(sm["fuel_burned_total_l"] - (40.0 - sm["fuel_l"])) < 0.3


def test_fuel_burned_total_counts_through_a_pit_stop():
    store, _ = run_sim(lap_seconds=3, total_laps=6, start_fuel=40.0, fuel_per_lap=7.0, refuel_lap=3)
    sm = store.summary()
    # the refuel (back to full) must not reduce what was burned: 5.x laps at 7 L each
    assert sm["fuel_burned_total_l"] > 33.0
    assert sm["fuel_burned_total_l"] > 40.0 - sm["fuel_l"] + 5.0  # more than the naive start-minus-now


def test_fuel_used_this_lap_so_far():
    store, _ = run_sim(lap_seconds=3, total_laps=3, start_fuel=40.0, fuel_per_lap=7.0)
    used = store.summary()["fuel_used_this_lap_so_far_l"]
    assert 0 <= used <= 7.5


def sim_frames(**kw):
    async def go():
        return [f async for f in simulator.frames(realtime=False, **kw)]

    return asyncio.run(go())


def test_laps_are_tracked_when_gt7_does_not_set_the_on_track_bit():
    """Real PS5 data: flags 0x0418 (in gear, turbo, ASM) while racing, bit 0 clear."""
    import dataclasses

    frames = [dataclasses.replace(f, flags=0x0418) for f in sim_frames(lap_seconds=3, total_laps=4)]
    assert not frames[0].car_on_track_bit and frames[0].on_track
    store, events = StateStore(), []
    for f in frames:
        ev = store.update(f)
        if ev:
            events.append(ev)
    assert [e.record.lap for e in events] == [1, 2, 3]


def test_paused_and_loading_frames_are_ignored():
    import dataclasses

    frames = sim_frames(lap_seconds=3, total_laps=3)
    paused = [dataclasses.replace(f, flags=f.flags | 0b10, lap=9, fuel=1.0) for f in frames[200:260]]  # bogus data
    store = StateStore()
    for f in frames[:200] + paused + frames[260:]:
        store.update(f)
    assert 9 not in store.fuel_at_lap_start and len(store.laps) == 2


def test_joining_mid_race_does_not_poison_fuel_stats():
    frames = [f for f in sim_frames(lap_seconds=3, total_laps=6, start_fuel=100.0, fuel_per_lap=7.0) if f.lap >= 3]
    store, events = StateStore(), []
    for f in frames:
        ev = store.update(f)
        if ev:
            events.append(ev)
    first, rest = events[0].record, [e.record for e in events[1:]]
    assert first.lap == 3 and not first.burn_valid  # we only saw part of lap 3
    assert 3 not in store.fuel_at_lap_start  # its start fuel is when we joined, not the real start
    assert all(r.burn_valid for r in rest) and rest
    assert abs(store.fuel_per_lap - 7.0) < 0.3  # computed from complete laps only


def test_lap_zero_grid_then_lap_one_records_lap_one_start_fuel():
    import dataclasses

    frames = sim_frames(lap_seconds=3, total_laps=3, start_fuel=40.0)
    grid = [dataclasses.replace(f, lap=0, speed=0.0) for f in frames[:90]]  # lap counter 0 before the start
    store = StateStore()
    for f in grid + frames:
        store.update(f)
    assert abs(store.fuel_at_lap_start[1] - 40.0) < 0.3  # recorded even though the race began at lap 0
    assert 0 not in store.fuel_at_lap_start


def test_a_single_garbage_upward_lap_number_does_not_freeze_tracking():
    import dataclasses

    frames = sim_frames(lap_seconds=3, total_laps=4)
    glitch = [dataclasses.replace(f, lap=9) for f in frames[200:206]]
    store, events = StateStore(), []
    for f in frames[:200] + glitch + frames[206:]:
        ev = store.update(f)
        if ev:
            events.append(ev)
    assert [e.record.lap for e in events] == [1, 2, 3]  # real laps still counted, glitch never became one
    assert max(store.fuel_at_lap_start) == 4


def test_full_race_still_reports_race_start_fuel():
    store, _ = run_sim(lap_seconds=3, total_laps=4, start_fuel=40.0)
    sm = store.summary()
    assert sm["fuel_at_race_start_l"] == 40.0 and "fuel_at_race_start_note" not in sm  # seen, not assumed


def test_joining_mid_race_in_the_sim_assumes_full_tank():
    frames = [f for f in sim_frames(lap_seconds=3, total_laps=6, start_fuel=100.0, fuel_per_lap=7.0) if f.lap >= 3]
    store = StateStore()
    for f in frames:
        store.update(f)
    sm = store.summary()
    assert sm["fuel_at_race_start_l"] == 100.0 and "assumed a full tank" in sm["fuel_at_race_start_note"]
    assert abs(sm["fuel_burned_total_l"] - (100.0 - sm["fuel_l"])) < 0.6
