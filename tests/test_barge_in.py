import asyncio
import threading
import time

import numpy as np
import pytest
import sounddevice as sd

from gt7_engineer import app
from gt7_engineer.audio import stt
from gt7_engineer.audio.tts import play_stream
from gt7_engineer.bus import Bus
from gt7_engineer.config import Config
from gt7_engineer.engineer import RaceEngineer
from gt7_engineer.state import StateStore
from gt7_engineer.telemetry import simulator


def test_play_stream_stops_midway_and_aborts(monkeypatch):
    writes, aborted = [], []
    stop = threading.Event()

    class FakeStream:
        def __init__(self, **kw): pass
        def start(self): pass
        def stop(self): pass
        def close(self): pass
        def abort(self): aborted.append(True)
        def write(self, c):
            writes.append(len(c))
            if len(writes) == 3:
                stop.set()  # the driver interrupts while the third chunk is playing
            return False

    monkeypatch.setattr(sd, "OutputStream", FakeStream)

    def gen():
        for _ in range(200):
            yield np.ones(4800, dtype=np.float32)

    play_stream(gen(), prebuffer=0.1, stop=stop)
    assert 3 <= len(writes) <= 5  # stopped right after the interrupt, not after all 200 chunks
    assert aborted  # buffered audio was discarded, not drained


def test_play_stream_interrupted_before_any_audio_plays_nothing(monkeypatch):
    opened = []

    class FakeStream:
        def __init__(self, **kw): opened.append(1)
        def start(self): pass
        def stop(self): pass
        def close(self): pass
        def abort(self): pass
        def write(self, c): raise AssertionError("must not write after an interrupt")

    monkeypatch.setattr(sd, "OutputStream", FakeStream)
    stop = threading.Event()
    stop.set()
    play_stream((np.ones(4800, dtype=np.float32) for _ in range(5)), prebuffer=0.1, stop=stop)


def make_engineer(stop):
    store = StateStore()
    said = []

    async def speak(t):
        said.append(t)
        await asyncio.sleep(0.3)

    return store, RaceEngineer(store, Bus(), None, speak, stop_speaking=stop), said


async def test_engineer_interrupt_stops_speech_and_notes_it():
    stopped = []
    store, eng, said = make_engineer(lambda: stopped.append(1))
    task = asyncio.create_task(eng._say("Fuel is critical, two laps left but three to run."))
    await asyncio.sleep(0.05)
    eng.interrupt()
    await task
    assert stopped == [1]
    assert eng.history[-1]["content"].endswith("[cut off by the driver]")
    eng.interrupt()  # nothing playing now: no second stop
    assert stopped == [1]


class FakeInput:
    """A microphone that plays a prepared signal."""

    audio = np.zeros(1)

    def __init__(self, callback, blocksize, **kw):
        self.cb, self.bs = callback, blocksize

    def __enter__(self):
        def feed():
            a = FakeInput.audio
            for i in range(0, len(a), self.bs):
                blk = a[i : i + self.bs]
                self.cb(np.pad(blk, (0, self.bs - len(blk))).reshape(-1, 1).astype(np.float32), self.bs, None, None)
                time.sleep(0.01)

        threading.Thread(target=feed, daemon=True).start()
        return self

    def __exit__(self, *a):
        pass


def prepare_mic(monkeypatch):
    rng = np.random.default_rng(1)
    quiet = lambda s: (rng.standard_normal(int(s * 16000)) * 0.001).astype(np.float32)
    voice = (np.sin(np.linspace(0, 2 * np.pi * 220 * 1.5, int(1.5 * 16000))) * 0.1).astype(np.float32)
    FakeInput.audio = np.concatenate([quiet(1.2), voice, quiet(1.5), quiet(1.0)])
    monkeypatch.setattr(sd, "InputStream", FakeInput)
    monkeypatch.setattr(sd, "query_devices", lambda kind=None: {"name": "fake mic"})
    monkeypatch.setattr(stt.WhisperListener, "_transcribe", lambda self, audio: "hello engineer")


async def hear(listener, seconds=4.0):
    got = []

    async def consume():
        async for t in listener.utterances():
            got.append(t)

    task = asyncio.create_task(consume())
    await asyncio.sleep(seconds)
    task.cancel()
    return got


async def test_listener_barge_in_interrupts_and_still_transcribes(monkeypatch):
    prepare_mic(monkeypatch)
    speaking = asyncio.Event()
    speaking.set()  # the engineer is talking the whole time
    fired = []
    listener = stt.WhisperListener(speaking=speaking, on_barge_in=lambda: fired.append(1))
    got = await hear(listener)
    assert fired == [1]  # interrupted exactly once
    assert got == ["hello engineer"]  # and the driver's words were not lost


async def test_listener_speaker_mode_is_deaf_while_engineer_talks(monkeypatch):
    prepare_mic(monkeypatch)
    speaking = asyncio.Event()
    speaking.set()
    listener = stt.WhisperListener(speaking=speaking, on_barge_in=None)
    assert await hear(listener) == []


async def test_listener_keeps_reading_while_whisper_is_slow(monkeypatch):
    prepare_mic(monkeypatch)
    calls = []

    def slow(self, audio):
        calls.append(time.monotonic())
        time.sleep(1.5)
        return "slow words"

    monkeypatch.setattr(stt.WhisperListener, "_transcribe", slow)
    listener = stt.WhisperListener()
    start = time.monotonic()
    got = await hear(listener, seconds=5)
    assert got == ["slow words"]


async def test_end_to_end_interrupt_via_app(monkeypatch):
    """Driver speech during the engineer's speech reaches engineer.interrupt through app.run."""
    prepare_mic(monkeypatch)
    stopped, said = [], []
    speaking = asyncio.Event()
    interrupter = app.Interrupter()
    listener = stt.WhisperListener(speaking=speaking, on_barge_in=interrupter.fire)

    async def speak(t):
        said.append(t)
        await asyncio.sleep(3)  # long speech the driver talks over

    src = simulator.frames(realtime=False, lap_seconds=3, total_laps=4, start_fuel=14.0, fuel_per_lap=7.0)
    await app.run(
        Config(), src, speak, listener.utterances(), use_llm=False, speaking=speaking,
        stop_speaking=lambda: stopped.append(1), interrupter=interrupter,
    )
    assert said, "the engineer should have spoken"
