import asyncio
import dataclasses
import fcntl
import os
import pty
import select
import signal
import struct
import subprocess
import sys
import termios
import threading
import time

from gt7_engineer import app, ui
from gt7_engineer.bus import Assessment
from gt7_engineer.config import Config
from gt7_engineer.engineer import RaceEngineer
from gt7_engineer.state import NewRace, StateStore
from gt7_engineer.telemetry import simulator


async def race(**kw):
    return [f async for f in simulator.frames(realtime=False, **kw)]


def feed(store, frames):
    events = []
    for f in frames:
        ev = store.update(f)
        if ev is not None:
            events.append(ev)
    return events


def test_second_race_resets_state_automatically():
    a = asyncio.run(race(lap_seconds=3, total_laps=4, start_fuel=40.0, grid_position=9))
    b = asyncio.run(race(lap_seconds=3, total_laps=3, start_fuel=30.0, grid_position=5))
    store = StateStore()
    store.add_report("tyre_wear", "fronts sixty")
    events = feed(store, a)
    assert not any(isinstance(e, NewRace) for e in events)  # a normal race never triggers it
    assert store.start_position == 9 and len(store.laps) == 3

    events = feed(store, b)
    races = [e for e in events if isinstance(e, NewRace)]
    assert len(races) == 1 and "back to" in races[0].reason
    assert store.start_position == 5  # the new race's grid, not the old one
    assert sorted(store.fuel_at_lap_start) == [1, 2, 3]  # only the new race's laps
    assert abs(store.fuel_at_lap_start[1] - 30.0) < 0.6
    assert len(store.laps) == 2  # laps 1 and 2 of the new race completed
    assert store.latest_report("tyre_wear") is None  # driver reports belonged to the old race


def test_brief_lap_glitch_does_not_reset():
    frames = asyncio.run(race(lap_seconds=3, total_laps=4))
    store = StateStore()
    feed(store, frames[:400])
    assert store.lap >= 2
    laps_before = dict(store.fuel_at_lap_start)
    glitch = [dataclasses.replace(f, lap=0) for f in frames[400:405]]  # 5 frames: well under the threshold
    events = feed(store, glitch + frames[405:420])
    assert not any(isinstance(e, NewRace) for e in events)
    assert all(store.fuel_at_lap_start[k] == v for k, v in laps_before.items())


def test_manual_reset_clears_everything():
    store = StateStore()
    feed(store, asyncio.run(race(lap_seconds=3, total_laps=3)))
    store.add_report("position", "P2")
    assert store.laps and store.fuel_at_lap_start and store.position_at_lap_start
    store.reset()
    assert not store.laps and not store.fuel_at_lap_start and not store.position_at_lap_start
    assert store.car is None and store.start_position is None and not store.reports and not store.frames
    feed(store, asyncio.run(race(lap_seconds=3, total_laps=3)))  # and it works normally afterwards
    assert len(store.laps) == 2


async def test_engineer_reset_drops_old_race_baggage():
    async def speak(t): pass

    store = StateStore()
    eng = RaceEngineer(store, app.Bus(), None, speak)
    eng.pending["fuel"] = Assessment("fuel", "fuel", "urgent", "Fuel critical.")
    eng.history.append({"role": "user", "content": "old question"})
    eng.awaiting_tyre_reply = True
    eng._urgent_at["fuel"] = 99.0
    eng.reset()
    assert not eng.pending and not eng.history and not eng.awaiting_tyre_reply and not eng._urgent_at


class RecordingUI(ui.PlainUI):
    def __init__(self):
        self.lines = []

    def log(self, kind, text):
        self.lines.append((kind, text))


async def test_app_resets_on_new_race_and_on_request():
    said = []

    async def speak(t):
        said.append(t)

    a = await race(lap_seconds=3, total_laps=3, start_fuel=40.0)
    b = await race(lap_seconds=3, total_laps=3, start_fuel=40.0)
    request = asyncio.Event()

    async def source():
        for i, f in enumerate(a + b):
            if i == len(a) + 400:  # late in the second race the driver presses r
                request.set()
            yield f
            await asyncio.sleep(0)

    rec = RecordingUI()
    old = ui.ui
    ui.use(rec)
    try:
        store = await app.run(Config(), source(), speak, None, use_llm=False, reset_request=request)
    finally:
        ui.use(old)
    system = [t for k, t in rec.lines if k == "system"]
    assert sum("New race detected" in t for t in system) == 1  # race two started: lap counter went back
    assert sum("you pressed r" in t for t in system) == 1  # and the keypress reset it again
    assert not request.is_set()  # consumed
    # after the last reset only the tail of race two remains in the state
    assert len(store.laps) <= 1 and min(store.fuel_at_lap_start) >= 2 or not store.fuel_at_lap_start


def test_key_listener_dispatch():
    hits = []
    k = ui.KeyListener({"r": lambda: hits.append("r")})
    assert k.handle("r") and k.handle("R") and not k.handle("x")
    assert hits == ["r", "r"]


def test_pressing_r_on_a_real_terminal_resets_and_terminal_is_restored():
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 130, 0, 0))
    p = subprocess.Popen(
        [sys.executable, "-m", "gt7_engineer.cli", "run", "--simulate", "--no-audio", "--no-llm", "--laps", "20", "--lap-seconds", "3"],
        stdin=slave, stdout=slave, stderr=slave, close_fds=True,
    )
    os.close(slave)
    chunks = []

    def drain():
        while True:
            try:
                if select.select([master], [], [], 0.2)[0]:
                    chunks.append(os.read(master, 65536))
                elif p.poll() is not None:
                    return
            except OSError:
                return

    th = threading.Thread(target=drain, daemon=True)
    th.start()
    time.sleep(4)
    os.write(master, b"r")  # one key press, no Enter
    time.sleep(2)
    p.send_signal(signal.SIGINT)
    assert p.wait(timeout=5) == 0
    th.join(2)
    out = b"".join(chunks).decode(errors="replace")
    assert "Race state reset" in out
    assert "new race" in out  # the hint is shown on the dashboard
    assert "\x1b[?1049l" in out  # left the alternate screen
