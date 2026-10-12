"""Speech-to-text: microphone capture with an adaptive energy VAD, transcribed by local Whisper (MLX).

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
MIN_THRESHOLD = 0.004  # RMS floor; laptop mics are quiet (ambient ~0.001, speech 0.01-0.05)
NOISE_MULTIPLE = 4.0  # speech must be this many times louder than the measured room noise
CALIBRATE_BLOCKS = 10  # 1 s of ambient noise at startup
MIN_SPEECH_BLOCKS = 3  # ignore clicks/bumps shorter than 0.3 s
END_SILENCE_BLOCKS = 8  # 0.8 s of quiet ends an utterance
MAX_BLOCKS = 200  # 20 s cap
HINT_EVERY_BLOCKS = 100  # while idle, report the mic level every 10 s


class WhisperListener:
    def __init__(self, model: str = "mlx-community/whisper-small-mlx", mute: "asyncio.Event | None" = None):
        self.model = model
        self.mute = mute  # set while the engineer is speaking so we don't hear ourselves

    async def warmup(self) -> None:
        """Load (and on first run, download) the model now, not on the driver's first sentence."""
        import numpy as np

        await asyncio.get_running_loop().run_in_executor(
            None, self._transcribe, np.zeros(SAMPLE_RATE, dtype=np.float32)
        )

    async def utterances(self) -> AsyncIterator[str]:
        import numpy as np
        import sounddevice as sd

        q: queue.Queue = queue.Queue()

        def cb(indata, frames, time_info, status):
            q.put(indata[:, 0].copy())

        loop = asyncio.get_running_loop()
        device = sd.query_devices(kind="input")["name"]
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, blocksize=BLOCK, callback=cb):
            print(f"🎤 Measuring room noise on '{device}' (stay quiet for a second)...", flush=True)
            noise = []
            while len(noise) < CALIBRATE_BLOCKS:
                block = await loop.run_in_executor(None, q.get)
                noise.append(float(np.sqrt(np.mean(block**2))))
            threshold = max(MIN_THRESHOLD, NOISE_MULTIPLE * float(np.median(noise)))
            print(
                f"🎤 Listening. Speak any time (speech threshold {threshold:.4f}, room noise {np.median(noise):.4f}).\n"
                "   If it never reacts, check System Settings > Privacy & Security > Microphone for your terminal.",
                flush=True,
            )
            buf: list = []
            loud_blocks = quiet = idle = 0
            peak = 0.0
            while True:
                block = await loop.run_in_executor(None, q.get)
                if self.mute is not None and self.mute.is_set():
                    buf, loud_blocks, quiet = [], 0, 0
                    continue
                rms = float(np.sqrt(np.mean(block**2)))
                loud = rms > threshold
                if loud:
                    if not buf:
                        print("🎤 hearing you...", flush=True)
                    buf.append(block)
                    loud_blocks += 1
                    quiet = 0
                elif buf:
                    buf.append(block)
                    quiet += 1
                else:
                    peak = max(peak, rms)
                    idle += 1
                    if idle >= HINT_EVERY_BLOCKS:
                        print(f"🎤 (idle) loudest sound in the last 10 s: {peak:.4f}, speech needs > {threshold:.4f}", flush=True)
                        idle, peak = 0, 0.0
                if buf and (quiet >= END_SILENCE_BLOCKS or len(buf) >= MAX_BLOCKS):
                    audio = np.concatenate(buf)
                    enough = loud_blocks >= MIN_SPEECH_BLOCKS
                    buf, loud_blocks, quiet = [], 0, 0
                    if not enough:
                        continue  # a click or bump, not speech
                    print("🎤 transcribing...", flush=True)
                    text = await loop.run_in_executor(None, self._transcribe, audio)
                    if text:
                        print(f'📝 You said: "{text}"', flush=True)
                        yield text
                    else:
                        print("📝 (no words recognised)", flush=True)

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
