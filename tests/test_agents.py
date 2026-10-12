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


class FakeLLM:
    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    async def chat(self, messages, **kw):
        self.prompts.append(messages[-1]["content"])
        return self.reply


def test_lap_trace_publishes_coaching_and_compares_to_best():
    from gt7_engineer.agents.lap_trace import LapTraceAgent

    store, bus = StateStore(), Bus()
    llm = FakeLLM("Brake later into turn 3.")
    agent = LapTraceAgent(store, bus, llm)
    drive(store, [agent], lap_seconds=3, total_laps=4)
    msgs = drain(bus)
    assert len(msgs) == 3 and msgs[0].message == "Brake later into turn 3."
    assert "Best lap" not in llm.prompts[0] and "Best lap" in llm.prompts[-1]
    assert "kph" in llm.prompts[0]


def test_lap_trace_silent_when_nothing_notable():
    from gt7_engineer.agents.lap_trace import LapTraceAgent

    store, bus = StateStore(), Bus()
    drive(store, [LapTraceAgent(store, bus, FakeLLM("Nothing notable."))], lap_seconds=3, total_laps=3)
    assert drain(bus) == []


def test_jev_decision_request_and_response():
    import httpx

    from gt7_engineer.bus import CHOICES
    from gt7_engineer.config import ModelConfig
    from gt7_engineer.llm import JevDecision

    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        import json

        seen["url"], seen["body"] = str(req.url), json.loads(req.content)
        return httpx.Response(200, json={"answers": {"alert": {
            "type": "choice", "choice": "urgent", "confidence": 0.9,
            "probabilities": {"ignore": 0.0, "mention": 0.1, "urgent": 0.9}}}})

    cfg = ModelConfig(base_url="https://openrouter.ai/api/alpha", model="typesafe/jev-1.13", kind="jev")
    http = httpx.AsyncClient(base_url=cfg.base_url, transport=httpx.MockTransport(handler))
    jev = JevDecision(cfg, "ignore", http)
    out = asyncio.run(jev.decide("Fuel check", {"laps_of_fuel": 1.2}, CHOICES))
    assert out == "urgent" and jev.last_probabilities["urgent"] == 0.9
    assert seen["url"].endswith("/alpha/decisions")
    q = seen["body"]["questions"]["alert"]
    assert q["type"] == "choice" and set(q["criteria"]) == set(CHOICES)
    assert seen["body"]["state"] == {"laps_of_fuel": 1.2}


def test_jev_decision_falls_back_on_error():
    import httpx

    from gt7_engineer.bus import CHOICES
    from gt7_engineer.config import ModelConfig
    from gt7_engineer.llm import JevDecision

    http = httpx.AsyncClient(base_url="https://x/api", transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    jev = JevDecision(ModelConfig(kind="jev"), "ignore", http)
    assert asyncio.run(jev.decide("x", {}, CHOICES)) == "ignore"


def test_fuel_agent_reports_heavier_burn():
    store, bus = StateStore(), Bus()
    fuel = FuelAgent(store, bus, ThresholdDecision(fuel_rule))
    drive(store, [fuel], lap_seconds=3, total_laps=8, start_fuel=60.0, fuel_per_lap=5.0, burn_growth=0.15)
    msgs = drain(bus)
    assert msgs, "thin margin plus heavier burn should be worth a mention"
    last = msgs[-1]
    assert last.data["burn_trend"] == "heavier" and last.data["burn_change_per_lap_l"] > 0
    assert "getting heavier" in last.message and "last lap" in last.message
