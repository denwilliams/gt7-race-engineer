import asyncio

from gt7_engineer.agents.fuel import FuelAgent, fuel_rule
from gt7_engineer.agents.pace import PaceAgent, pace_rule
from gt7_engineer.bus import IGNORE, MENTION, URGENT, Bus
from gt7_engineer.llm import ThresholdDecision
from gt7_engineer.state import StateStore
from gt7_engineer.telemetry import simulator


def drive(store, agents, **kw):
    async def go():
        async for f in simulator.frames(realtime=False, **kw):
            ev = store.update(f)
            if ev:
                for a in agents:
                    await a.handle_lap(ev)

    asyncio.run(go())


def drain(bus):
    out = []
    while not bus.queue.empty():
        out.append(bus.queue.get_nowait())
    return out


def test_fuel_urgent_when_cannot_finish():
    store, bus = StateStore(), Bus()
    fuel = FuelAgent(store, bus, ThresholdDecision(fuel_rule))
    drive(store, [fuel], lap_seconds=3, total_laps=6, start_fuel=18.0, fuel_per_lap=7.0)
    levels = [a.level for a in drain(bus)]
    assert URGENT in levels


def test_fuel_quiet_when_comfortable():
    store, bus = StateStore(), Bus()
    fuel = FuelAgent(store, bus, ThresholdDecision(fuel_rule))
    drive(store, [fuel], lap_seconds=3, total_laps=4, start_fuel=100.0, fuel_per_lap=5.0)
    assert drain(bus) == []


def test_pace_mentions_best_lap():
    store, bus = StateStore(), Bus()
    pace = PaceAgent(store, bus, ThresholdDecision(pace_rule))
    drive(store, [pace], lap_seconds=3, total_laps=4)
    msgs = drain(bus)
    assert msgs and all(m.level == MENTION for m in msgs)
    assert any("Lap" in m.message for m in msgs)
