"""Text-to-speech, selected by `[tts] engine` in config.toml.

- say:    macOS `say` command. Zero setup, robotic.
- qwen3:  Qwen3-TTS 0.6B CustomVoice via mlx-audio. Streams, style-controllable.
- kokoro: Kokoro-82M via mlx-audio. Very fast.

Add an engine by implementing `speak(text)` (and optionally `warmup()`).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Protocol

from ..config import TTSConfig

log = logging.getLogger(__name__)


class TTS(Protocol):
    async def speak(self, text: str) -> None: ...


class SayTTS:
    def __init__(self, voice: str = "Samantha", speed: float = 1.0):
        self.voice, self.rate = voice or "Samantha", round(210 * speed)

    async def speak(self, text: str) -> None:
        proc = await asyncio.create_subprocess_exec("say", "-v", self.voice, "-r", str(self.rate), text)
        await proc.wait()

    async def warmup(self) -> None:
        pass


class PrintTTS:
    """Text-only output for development without audio."""

    async def speak(self, text: str) -> None:
        print(f"\n🎙  ENGINEER: {text}\n", flush=True)


SAMPLE_RATE = 24000
ENGINE_DEFAULTS = {
    "qwen3": {
        "model": "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit",
        "voice": "Ryan",
        "instruct": "Calm, clear, brisk race engineer on the radio.",
    },
    "kokoro": {"model": "mlx-community/Kokoro-82M-bf16", "voice": "am_adam", "instruct": ""},
}


class MLXTTS:
    """Neural TTS through mlx-audio (requires the `audio` extra). Audio is played as it is generated."""

    def __init__(self, cfg: TTSConfig):
        d = ENGINE_DEFAULTS[cfg.engine]
        self.engine = cfg.engine
        self.model_id = cfg.model or d["model"]
        self.voice = cfg.voice or d["voice"]
        self.instruct = cfg.instruct or d["instruct"]
        self.speed = cfg.speed
        self._model = None

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
                text=text, speaker=self.voice, language="English", instruct=self.instruct,
                stream=True, streaming_interval=0.32,
            )
        else:
            gen = m.generate(text=text, voice=self.voice, speed=self.speed, lang_code="a")
        for r in gen:
            yield np.asarray(r.audio, dtype=np.float32).reshape(-1)

    def _play(self, chunks) -> None:
        import sounddevice as sd

        with sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as out:
            for c in chunks:
                out.write(c.reshape(-1, 1))

    def _speak_blocking(self, text: str) -> None:
        self._play(self._chunks(text))

    async def speak(self, text: str) -> None:
        await asyncio.get_running_loop().run_in_executor(None, self._speak_blocking, text)

    async def warmup(self) -> None:
        """Load the model and run one silent generation: the first call is much slower than the rest."""

        def go() -> None:
            for _ in self._chunks("Radio check."):
                pass

        await asyncio.get_running_loop().run_in_executor(None, go)


def make_tts(cfg: TTSConfig) -> TTS:
    if cfg.engine == "say":
        return SayTTS(cfg.voice, cfg.speed)
    if cfg.engine in ENGINE_DEFAULTS:
        return MLXTTS(cfg)
    raise SystemExit(f"Unknown [tts] engine {cfg.engine!r}; use one of: say, {', '.join(ENGINE_DEFAULTS)}")
