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


def test_tts_engine_selection_and_playback_path():
    import numpy as np

    from gt7_engineer.audio.tts import MLXTTS, SayTTS, make_tts
    from gt7_engineer.config import TTSConfig

    assert isinstance(make_tts(TTSConfig(engine="say")), SayTTS)
    q, k = make_tts(TTSConfig(engine="qwen3")), make_tts(TTSConfig(engine="kokoro"))
    assert isinstance(q, MLXTTS) and "Qwen3-TTS" in q.model_id and q.voice == "Ryan"
    assert isinstance(k, MLXTTS) and "Kokoro" in k.model_id and k.voice == "am_adam"
    assert make_tts(TTSConfig(engine="kokoro", voice="af_heart")).voice == "af_heart"

    class R:  # stand-in for an mlx-audio result
        audio = np.ones(4)

    class FakeModel:
        def generate_custom_voice(self, **kw):
            assert kw["speaker"] == "Ryan" and kw["stream"] is True
            yield R()
            yield R()

        def generate(self, **kw):
            assert kw["voice"] == "am_adam" and kw["lang_code"] == "a"
            yield R()

    played = []
    for tts, n in ((q, 2), (k, 1)):
        tts._model = FakeModel()
        tts._play = lambda chunks, played=played: played.append(list(chunks))
        asyncio.run(tts.speak("hello"))
        assert len(played[-1]) == n and played[-1][0].dtype == np.float32


def test_chat_client_sends_reasoning_effort_and_handles_empty():
    import json

    import httpx

    from gt7_engineer.config import ModelConfig
    from gt7_engineer.llm import ChatClient

    seen = []

    def handler(req):
        seen.append(json.loads(req.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "", "reasoning": "hmm"}, "finish_reason": "length"}]})

    cfg = ModelConfig(base_url="http://x/v1", reasoning_effort="none")
    c = ChatClient(cfg)
    c._http = httpx.AsyncClient(base_url=cfg.base_url, transport=httpx.MockTransport(handler))
    assert asyncio.run(c.chat([{"role": "user", "content": "hi"}])) == ""
    assert seen[0]["reasoning_effort"] == "none"
    cfg2 = ModelConfig(base_url="http://x/v1")
    c2 = ChatClient(cfg2)
    c2._http = httpx.AsyncClient(base_url=cfg2.base_url, transport=httpx.MockTransport(handler))
    asyncio.run(c2.chat([{"role": "user", "content": "hi"}]))
    assert "reasoning_effort" not in seen[1]


def test_play_stream_overlaps_generation_and_propagates_errors(monkeypatch):
    import threading

    import numpy as np
    import sounddevice as sd

    from gt7_engineer.audio.tts import play_stream

    writes, gen_thread = [], []

    class FakeStream:
        def __init__(self, **kw): pass
        def start(self): pass
        def stop(self): pass
        def close(self): pass
        def write(self, c):
            writes.append((threading.current_thread().name, len(c)))
            return False

    monkeypatch.setattr(sd, "OutputStream", FakeStream)

    def gen():
        gen_thread.append(threading.current_thread().name)
        for _ in range(5):
            yield np.ones(4800, dtype=np.float32)  # 0.2 s each

    assert play_stream(gen(), prebuffer=0.5) == 0.0
    assert len(writes) == 5 and sum(n for _, n in writes) == 24000
    assert gen_thread[0] != writes[0][0]  # generation is not on the playback thread

    def boom():
        yield np.ones(4800, dtype=np.float32)
        raise RuntimeError("model failed")

    import pytest

    with pytest.raises(RuntimeError, match="model failed"):
        play_stream(boom(), prebuffer=0.1)
