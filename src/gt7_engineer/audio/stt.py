"""Speech-to-text: microphone capture with a simple energy VAD, transcribed by local Whisper (MLX).

Requires the `audio` extra:  uv sync --extra audio
"""

from __future__ import annotations

import asyncio
import logging
import queue
from collections.abc import AsyncIterator

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
BLOCK = 1600  # 100 ms
SPEECH_RMS = 0.015
END_SILENCE_BLOCKS = 8  # 0.8 s of quiet ends an utterance
MAX_BLOCKS = 200  # 20 s cap


class WhisperListener:
    def __init__(self, model: str = "mlx-community/whisper-small-mlx", mute: "asyncio.Event | None" = None):
        self.model = model
        self.mute = mute  # set while the engineer is speaking so we don't hear ourselves

    async def utterances(self) -> AsyncIterator[str]:
        import numpy as np
        import sounddevice as sd

        q: queue.Queue = queue.Queue()

        def cb(indata, frames, time_info, status):
            q.put(indata[:, 0].copy())

        loop = asyncio.get_running_loop()
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, blocksize=BLOCK, callback=cb):
            buf: list = []
            quiet = 0
            while True:
                block = await loop.run_in_executor(None, q.get)
                if self.mute is not None and self.mute.is_set():
                    buf, quiet = [], 0
                    continue
                loud = float(np.sqrt(np.mean(block**2))) > SPEECH_RMS
                if loud:
                    buf.append(block)
                    quiet = 0
                elif buf:
                    buf.append(block)
                    quiet += 1
                if buf and (quiet >= END_SILENCE_BLOCKS or len(buf) >= MAX_BLOCKS):
                    audio = np.concatenate(buf)
                    buf, quiet = [], 0
                    text = await loop.run_in_executor(None, self._transcribe, audio)
                    if text:
                        yield text

    def _transcribe(self, audio) -> str:
        import mlx_whisper

        out = mlx_whisper.transcribe(audio, path_or_hf_repo=self.model, language="en")
        return out["text"].strip()


async def stdin_utterances() -> AsyncIterator[str]:
    """Type to the engineer instead of speaking (for development)."""
    loop = asyncio.get_running_loop()
    while True:
        line = await loop.run_in_executor(None, input)
        yield line
