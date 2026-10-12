"""Text-to-speech, selected by `[tts] engine` in config.toml.

- say:    macOS `say` command. Zero setup, robotic.
- qwen3:  Qwen3-TTS 0.6B CustomVoice via mlx-audio. Streams. The 0.6B model ignores or is slowed by `instruct`; only 1.7B models are meant to follow it.
- kokoro: Kokoro-82M via mlx-audio. Very fast.

Add an engine by implementing `speak(text)` (and optionally `warmup()`).
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Protocol

from ..config import TTSConfig

log = logging.getLogger(__name__)


class TTS(Protocol):
    async def speak(self, text: str) -> None: ...

    def stop(self) -> None:
        """Cut off the current utterance immediately (driver interrupted)."""


class SayTTS:
    def __init__(self, voice: str = "Samantha", speed: float = 1.0):
        self.voice, self.rate = voice or "Samantha", round(210 * speed)

    _proc = None

    async def speak(self, text: str) -> None:
        self._proc = await asyncio.create_subprocess_exec("say", "-v", self.voice, "-r", str(self.rate), text)
        await self._proc.wait()

    def stop(self) -> None:
        if self._proc is not None and self._proc.returncode is None:
            self._proc.terminate()

    async def warmup(self) -> None:
        pass


class PrintTTS:
    """Text-only output for development without audio."""

    def stop(self) -> None:
        pass

    async def speak(self, text: str) -> None:
        print(f"\n🎙  ENGINEER: {text}\n", flush=True)


SAMPLE_RATE = 24000
WARMUP_LINES = [
    "Box.",
    "Lap three, one thirty two point four. Plus point two to your best.",
    "Fuel is critical. Two laps left but three to run. Short by one lap, so lift and coast to save fuel, or pit now.",
]
ENGINE_DEFAULTS = {
    "qwen3": {
        "model": "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit",
        "voice": "Ryan",
        "instruct": "",
    },
    "kokoro": {"model": "mlx-community/Kokoro-82M-bf16", "voice": "am_adam", "instruct": ""},
}


def play_stream(chunks, prebuffer: float = 0.6, stop=None, on_stream=None) -> float:
    """Play audio chunks as they are generated; returns seconds of underrun (audible gaps).

    Generation runs in its own thread and this thread only does blocking
    `write()` calls, so no Python runs in the audio path. (Generating and playing
    in one thread, or feeding the device from a Python callback, produced audible
    stutter at the start of every phrase.) Playback starts once `prebuffer`
    seconds are queued, or when generation finishes.

    `stop` (a threading.Event) ends playback early: generation is abandoned and the
    sound card's buffer is discarded so audio cuts out at once. `on_stream` receives the
    open stream so another thread can abort it.
    """
    import queue
    import threading

    import numpy as np
    import sounddevice as sd

    q: queue.Queue = queue.Queue()
    error: list[BaseException] = []

    def produce() -> None:
        try:
            for c in chunks:
                if stop is not None and stop.is_set():
                    break
                q.put(np.asarray(c, dtype=np.float32).reshape(-1, 1))
        except BaseException as e:  # surface generation failures to the caller
            error.append(e)
        finally:
            q.put(None)

    threading.Thread(target=produce, daemon=True).start()

    pending, queued, ended = [], 0, False
    while queued < prebuffer * SAMPLE_RATE and not (stop is not None and stop.is_set()):
        c = q.get()
        if c is None:
            ended = True
            break
        pending.append(c)
        queued += len(c)

    underflows = 0
    stopped = lambda: stop is not None and stop.is_set()  # noqa: E731
    if pending and not stopped():
        out = sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", latency="high")
        if on_stream:
            on_stream(out)
        out.start()
        try:
            for c in pending:
                if stopped():
                    break
                underflows += bool(out.write(c))
            while not ended and not stopped():
                c = q.get()
                if c is None:
                    break
                underflows += bool(out.write(c))
        except sd.PortAudioError:
            if not stopped():  # an abort() from stop() surfaces here; anything else is real
                raise
        finally:
            if stopped():
                out.abort()  # discard buffered audio
            else:
                out.stop()
            out.close()
    if error and not stopped():
        raise error[0]
    if underflows:
        log.warning("TTS playback underflowed %d time(s); try a larger prebuffer", underflows)
    return float(underflows)


class MLXTTS:
    """Neural TTS through mlx-audio (requires the `audio` extra). Audio is played as it is generated."""

    def __init__(self, cfg: TTSConfig):
        d = ENGINE_DEFAULTS[cfg.engine]
        self.engine = cfg.engine
        self.model_id = cfg.model or d["model"]
        self.voice = cfg.voice or d["voice"]
        self.instruct = cfg.instruct or d["instruct"]
        self.speed = cfg.speed
        self.prebuffer = cfg.prebuffer
        self.last_underrun_s = 0.0  # device underflow count for the last utterance (0 = clean)
        self._model = None
        self._stop = threading.Event()
        self._out = None  # the open output stream, so stop() can abort it

    def _load(self):
        if self._model is None:
            from mlx_audio.tts.utils import load_model

            self._model = load_model(self.model_id)
        return self._model

    def _chunks(self, text: str):
        import numpy as np

        m = self._load()
        if self.engine == "qwen3":
            gen = m.generate_custom_voice(
                text=text, speaker=self.voice, language="English", instruct=self.instruct or None,
                stream=True,
            )
        else:
            gen = m.generate(text=text, voice=self.voice, speed=self.speed, lang_code="a")
        for r in gen:
            yield np.asarray(r.audio, dtype=np.float32).reshape(-1)

    def _play(self, chunks) -> None:
        self.last_underrun_s = play_stream(chunks, self.prebuffer, self._stop, lambda out: setattr(self, "_out", out))

    def stop(self) -> None:
        self._stop.set()
        out = self._out
        if out is not None:
            try:
                out.abort()  # unblocks a write() in progress and drops buffered audio
            except Exception:
                pass

    def _speak_blocking(self, text: str) -> None:
        self._play(self._chunks(text))

    async def speak(self, text: str) -> None:
        self._stop.clear()
        self._out = None
        await asyncio.get_running_loop().run_in_executor(None, self._speak_blocking, text)

    async def warmup(self) -> None:
        """Load the model and run silent generations of a few lengths.

        The first call at a new length can be much slower (Kokoro: ~2.7 s vs 0.2 s), so cover short and long.
        """

        def go() -> None:
            for text in WARMUP_LINES:
                for _ in self._chunks(text):
                    pass

        await asyncio.get_running_loop().run_in_executor(None, go)


def make_tts(cfg: TTSConfig) -> TTS:
    if cfg.engine == "say":
        return SayTTS(cfg.voice, cfg.speed)
    if cfg.engine in ENGINE_DEFAULTS:
        return MLXTTS(cfg)
    raise SystemExit(f"Unknown [tts] engine {cfg.engine!r}; use one of: say, {', '.join(ENGINE_DEFAULTS)}")
