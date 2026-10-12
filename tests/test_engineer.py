import asyncio

from gt7_engineer import app
from gt7_engineer.bus import Assessment, Bus, URGENT
from gt7_engineer.config import Config
from gt7_engineer.engineer import RaceEngineer, TYRE_QUESTION
from gt7_engineer.state import StateStore
from gt7_engineer.telemetry import simulator


class FakeClient:
    def __init__(self, reply="Copy."):
        self.reply, self.calls = reply, []

    async def chat(self, messages, **kw):
        self.calls.append(messages)
        return self.reply


def make(client=None):
    said = []

    async def speak(t):
        said.append(t)

    store = StateStore()
    return store, Bus(), said, speak


async def prime(store):
    async for f in simulator.frames(realtime=False, lap_seconds=2, total_laps=2):
        store.update(f)


async def test_driver_tyre_report_is_stored():
    store, bus, said, speak = make()
    await prime(store)
    eng = RaceEngineer(store, bus, None, speak)
    eng.awaiting_tyre_reply = True
    await eng.on_driver("fronts are about sixty percent")
    assert store.latest_report("tyre_wear").value.startswith("fronts")
    assert said and "tyres noted" in said[0]


async def test_driver_question_gets_state_in_prompt():
    store, bus, said, speak = make()
    await prime(store)
    client = FakeClient("You have plenty of fuel.")
    eng = RaceEngineer(store, bus, client, speak)
    await eng.on_driver("how's my fuel?")
    assert said == ["You have plenty of fuel."]
    assert "fuel_l" in client.calls[0][0]["content"]


async def test_urgent_spoken_immediately_normal_queued():
    store, bus, said, speak = make()
    await prime(store)
    eng = RaceEngineer(store, bus, None, speak)
    task = asyncio.create_task(eng.run())
    bus.queue.put_nowait(Assessment("pace", "pace", "mention", "Lap slow."))
    bus.queue.put_nowait(Assessment("fuel", "fuel", URGENT, "Fuel critical."))
    await asyncio.sleep(0.1)
    task.cancel()
    assert said == ["Fuel critical."]
    assert "pace" in eng.pending


async def test_lap_ping_combines_and_asks_tyres():
    store, bus, said, speak = make()
    await prime(store)
    eng = RaceEngineer(store, bus, None, speak, tyre_ask_every=1)
    eng.pending["pace"] = Assessment("pace", "pace", "mention", "Lap 1: 0:30.000.")

    class Ev:  # minimal NewLap stand-in
        record = store.laps[-1] if store.laps else type("R", (), {"lap": 1})()

    await eng.on_lap(Ev())
    assert said and "0:30.000" in said[0] and TYRE_QUESTION in said[0]
    assert eng.awaiting_tyre_reply


async def test_end_to_end_simulated_race_without_llm():
    said = []

    async def speak(t):
        said.append(t)

    src = simulator.frames(realtime=False, lap_seconds=4, total_laps=4, start_fuel=22.0, fuel_per_lap=7.0)
    store = await app.run(Config(), src, speak, None, use_llm=False)
    assert len(store.laps) == 3
    text = " ".join(said)
    assert "Lap" in text  # lap updates reached the driver
    assert "Fuel" in text  # fuel shortfall flagged


async def test_listener_crash_is_reported_not_swallowed(caplog):
    async def broken():
        raise RuntimeError("no microphone")
        yield ""

    said = []

    async def speak(t):
        said.append(t)

    src = simulator.frames(realtime=False, lap_seconds=2, total_laps=2)
    with caplog.at_level("ERROR"):
        await app.run(Config(), src, speak, broken(), use_llm=False)
    assert any("can no longer hear you" in r.getMessage() for r in caplog.records)


async def test_driver_text_from_listener_reaches_engineer():
    said = []

    async def speak(t):
        said.append(t)

    async def heard():
        yield "fronts are at sixty percent tyre wear"
        await asyncio.sleep(0.2)

    src = simulator.frames(realtime=False, lap_seconds=2, total_laps=3)
    store = await app.run(Config(), src, speak, heard(), use_llm=False)
    assert store.latest_report("tyre_wear") is not None


async def test_tyre_statements_vs_questions():
    store, bus, said, speak = make()
    await prime(store)
    eng = RaceEngineer(store, bus, FakeClient("ok"), speak)
    await eng.on_driver("how are the tyres looking?")
    assert store.latest_report("tyre_wear") is None  # a question, not a report
    await eng.on_driver("fronts are at sixty percent, rears seventy")
    assert store.latest_report("tyre_wear").value.startswith("fronts")


async def test_microphone_keeps_reading_while_engineer_is_busy():
    import time

    resumed = []
    said = []

    async def slow_speak(t):
        said.append(t)
        await asyncio.sleep(0.6)  # speaking takes a while

    async def heard():
        yield "how is my fuel"
        resumed.append(time.monotonic())  # generator resumed = mic loop reading again
        yield "and the tyres"

    start = time.monotonic()
    src = simulator.frames(realtime=False, lap_seconds=2, total_laps=3)
    await app.run(Config(), src, slow_speak, heard(), use_llm=False)
    assert resumed and resumed[0] - start < 0.3  # not blocked behind the 0.6 s reply
