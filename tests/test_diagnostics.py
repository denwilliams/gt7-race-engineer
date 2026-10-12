import asyncio
import json

from gt7_engineer import app, ui
from gt7_engineer.agents.lap_trace import LapTraceAgent
from gt7_engineer.bus import Bus
from gt7_engineer.config import Config
from gt7_engineer.engineer import RaceEngineer
from gt7_engineer.recorder import Recorder
from gt7_engineer.state import StateStore
from gt7_engineer.telemetry import simulator


async def frames(**kw):
    return [f async for f in simulator.frames(realtime=False, **kw)]


class RecordingUI(ui.PlainUI):
    def __init__(self):
        self.lines = []

    def log(self, kind, text):
        self.lines.append((kind, text))


async def test_failing_lap_update_is_reported_not_swallowed(monkeypatch, caplog):
    async def boom(self, event):
        raise RuntimeError("lap update exploded")

    monkeypatch.setattr(RaceEngineer, "on_lap", boom)

    async def speak(t):
        pass

    src = simulator.frames(realtime=False, lap_seconds=2, total_laps=3)
    with caplog.at_level("ERROR"):
        await app.run(Config(), src, speak, None, use_llm=False)
    assert any("background task failed" in r.getMessage() and "exploded" in r.getMessage() for r in caplog.records)


async def test_every_lap_counter_change_is_logged_before_any_logic():
    rec = RecordingUI()
    old = ui.ui
    ui.use(rec)
    try:
        async def speak(t):
            pass

        await app.run(Config(), simulator.frames(realtime=False, lap_seconds=2, total_laps=3), speak, None, use_llm=False)
    finally:
        ui.use(old)
    lines = [t for k, t in rec.lines if k == "system" and t.startswith("lap counter")]
    assert len(lines) == 3 and "None -> 1" in lines[0] and "1 -> 2" in lines[1] and "2 -> 3" in lines[2]
    assert "running: True" in lines[1] and "flags 0x" in lines[1]


async def test_recorder_writes_jsonl_decimated_but_keeps_transitions(tmp_path):
    path = tmp_path / "race.jsonl"
    r = Recorder(str(path))
    fs = await frames(lap_seconds=3, total_laps=3)  # ~540 frames at 60 Hz
    for f in fs:
        r.write(f)
    r.close()
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    assert 20 < len(rows) < len(fs) / 4  # ~10 Hz, not 60 Hz
    assert {1, 2, 3} <= {x["lap"] for x in rows}  # every lap change is captured
    for key in ("t", "lap", "laps", "flags", "on_track", "pos", "cars", "fuel", "last_ms", "best_ms", "x", "y", "z", "ms"):
        assert key in rows[0]


def test_lap_trace_ignores_non_answers():
    from gt7_engineer.agents.lap_trace import is_non_answer

    for non in ("", "Nothing notable.", "No clear improvement stands out.", "No notable differences.", "nothing stands out"):
        assert is_non_answer(non), non
    assert not is_non_answer("Brake later into turn three.")


async def test_lap_updates_are_spoken_with_the_flags_a_real_ps5_sends():
    """Regression: real GT7 flags were 0x0418 while racing (bit 0 clear); the engineer used to stay silent."""
    import dataclasses

    said = []

    async def speak(t):
        said.append(t)

    async def source():
        async for f in simulator.frames(realtime=False, lap_seconds=3, total_laps=4):
            yield dataclasses.replace(f, flags=0x0418)

    store = await app.run(Config(), source(), speak, None, use_llm=False)
    assert len(store.laps) == 3
    updates = [t for t in said if t.startswith("Lap ")]
    assert len(updates) == 3 and updates[0].startswith("Lap two of four.")
