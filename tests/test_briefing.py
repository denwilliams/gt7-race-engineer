import asyncio

from gt7_engineer.briefing import lap_update, say_delta, say_int, say_lap_time, say_num
from gt7_engineer.state import StateStore
from gt7_engineer.telemetry import simulator


def test_spoken_numbers():
    assert say_int(32) == "thirty-two" and say_int(0) == "zero" and say_int(100) == "one hundred"
    assert say_num(1.7) == "one point seven" and say_num(12.0) == "twelve" and say_num(-1.3) == "minus one point three"
    assert say_lap_time(92400) == "one thirty-two point four"
    assert say_lap_time(65050) == "one oh five point one"
    assert say_lap_time(5200) == "five point two"
    assert say_delta(300) == "three tenths" and say_delta(100) == "one tenth"
    assert say_delta(1200) == "one point two seconds" and say_delta(20) == ""


def run(**kw):
    store, events = StateStore(), []

    async def go():
        async for f in simulator.frames(realtime=False, **kw):
            ev = store.update(f)
            if ev:
                events.append((ev, lap_update(store, ev)))

    asyncio.run(go())
    return store, events


def test_update_contents_across_a_race():
    store, events = run(lap_seconds=3, total_laps=6, start_fuel=40.0, fuel_per_lap=7.0)
    first = events[0][1]
    assert first.startswith("Lap two of six.") and "Last lap" in first and "Fuel" in first
    assert "off best" not in first and "New best" not in first  # nothing to compare with yet
    texts = [t for _, t in events]
    assert any("off best" in t or "New best" in t or "level with" in t for t in texts[1:])
    assert all("Lap " in t and " of six." in t for t in texts)
    assert "Short, save fuel or box." in texts[-1]  # 40 L, 7 L/lap, 6 laps: cannot finish


def test_update_burn_trend_and_driver_position():
    store, events = run(lap_seconds=3, total_laps=8, start_fuel=200.0, burn_growth=0.15)
    assert any("Burn rising." in t for _, t in events)
    store.add_report("position", "P4 at lap 3")
    assert "You said P4." in lap_update(store, events[-1][0])


def test_update_is_short():
    _, events = run(lap_seconds=3, total_laps=6, start_fuel=40.0)
    for _, t in events:
        assert len(t.split()) <= 30, t
