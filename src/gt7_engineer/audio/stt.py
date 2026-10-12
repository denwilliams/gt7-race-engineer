"""Speech-to-text: microphone capture with an adaptive energy VAD, transcribed by local Whisper (MLX).

Requires the `audio` extra:  uv sync --extra audio
"""

from __future__ import annotations

import asyncio
import logging
import queue
import time
from collections.abc import AsyncIterator

from .. import ui

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
BLOCK = 1600  # 100 ms
MIN_THRESHOLD = 0.004  # RMS floor; laptop mics are quiet (ambient ~0.001, speech 0.01-0.05)
NOISE_MULTIPLE = 4.0  # speech must be this many times louder than the measured room noise
CALIBRATE_BLOCKS = 10  # 1 s of ambient noise at startup
MIN_SPEECH_BLOCKS = 3  # ignore clicks/bumps shorter than 0.3 s
INTERRUPT_BLOCKS = 3  # speech this long (0.3 s) over the engineer interrupts it
END_SILENCE_BLOCKS = 8  # 0.8 s of quiet ends an utterance
MAX_BLOCKS = 200  # 20 s cap
HINT_EVERY_BLOCKS = 100  # while idle, report the mic level every 10 s
ECHO_TAIL_S = 0.5  # keep ignoring the mic this long after the engineer stops (room echo)


async def _next_block(q: "queue.Queue"):
    """Next mic block. Waits in short slices so no worker thread is ever stuck forever
    (a stuck thread makes the process hang on exit after the race ends or the task is cancelled)."""
    loop = asyncio.get_running_loop()
    while True:
        try:
            return await loop.run_in_executor(None, lambda: q.get(timeout=0.2))
        except queue.Empty:
            continue


class WhisperListener:
    def __init__(
        self,
        model: str = "mlx-community/whisper-small-mlx",
        speaking: "asyncio.Event | None" = None,
        on_barge_in=None,
    ):
        self.model = model
        self.mute = speaking  # set while the engineer is speaking
        # None: ignore the mic while the engineer speaks (speakers; avoids hearing itself).
        # callable: keep the mic open (headphones) and call it when the driver starts talking over the engineer.
        self.on_barge_in = on_barge_in
        self._tx_lock = asyncio.Lock()

    async def warmup(self) -> None:
        """Load (and on first run, download) the model now, not on the driver's first sentence."""
        import numpy as np

        await asyncio.get_running_loop().run_in_executor(
            None, self._transcribe, np.zeros(SAMPLE_RATE, dtype=np.float32)
        )

    async def utterances(self) -> AsyncIterator[str]:
        """Yield what the driver says. Capture and transcription run in the background,
        so the microphone keeps being read while Whisper works."""
        out: asyncio.Queue = asyncio.Queue()
        capture = asyncio.create_task(self._capture(out))
        try:
            while True:
                item = await out.get()
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            capture.cancel()

    async def _transcribe_into(self, audio, out: asyncio.Queue) -> None:
        loop = asyncio.get_running_loop()
        async with self._tx_lock:  # one transcription at a time, in order
            try:
                text = await loop.run_in_executor(None, self._transcribe, audio)
            except Exception as e:  # one bad transcription must not end listening
                log.exception("transcription failed")
                ui.log("warn", f"transcription failed ({e}); still listening")
                return
        ui.set_mic("listening")
        if text:
            ui.log("you", text)
            out.put_nowait(text)
        else:
            ui.log("system", "(no words recognised)")

    async def _capture(self, out: asyncio.Queue) -> None:
        try:
            await self._capture_loop(out)
        except asyncio.CancelledError:
            raise
        except BaseException as e:
            out.put_nowait(e)

    async def _capture_loop(self, out: asyncio.Queue) -> None:
        import numpy as np
        import sounddevice as sd

        q: queue.Queue = queue.Queue()

        def cb(indata, frames, time_info, status):
            q.put(indata[:, 0].copy())

        loop = asyncio.get_running_loop()
        device = sd.query_devices(kind="input")["name"]
        tasks: set[asyncio.Task] = set()
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, blocksize=BLOCK, callback=cb):
            ui.set_mic("measuring room noise (stay quiet for a second)")
            noise = []
            while len(noise) < CALIBRATE_BLOCKS:
                block = await _next_block(q)
                noise.append(float(np.sqrt(np.mean(block**2))))
            threshold = max(MIN_THRESHOLD, NOISE_MULTIPLE * float(np.median(noise)))
            mode = "mic stays open while the engineer talks; you can interrupt" if self.on_barge_in else "mic is muted while the engineer talks"
            ui.log(
                "system",
                f"Listening on '{device}' (speech threshold {threshold:.4f}, room noise {np.median(noise):.4f}; {mode}). "
                "If it never reacts, check System Settings > Privacy & Security > Microphone for your terminal.",
            )
            mic_state = ""

            def mic(status: str, show: bool = True) -> None:
                nonlocal mic_state
                if status != mic_state:
                    mic_state = status
                    ui.set_mic(status, show)

            mic("listening")
            buf: list = []
            loud_blocks = quiet = idle = 0
            peak = 0.0
            barged = False
            last_muted = float("-inf")
            while True:
                block = await _next_block(q)
                speaking = self.mute is not None and self.mute.is_set()
                if speaking and self.on_barge_in is None:  # speaker mode: deaf while it talks
                    buf, loud_blocks, quiet = [], 0, 0
                    last_muted = time.monotonic()
                    mic("muted (engineer talking)")
                    continue
                if self.on_barge_in is None and time.monotonic() - last_muted < ECHO_TAIL_S:
                    continue
                rms = float(np.sqrt(np.mean(block**2)))
                loud = rms > threshold
                if loud:
                    if not buf:
                        mic("hearing you")
                    buf.append(block)
                    loud_blocks += 1
                    quiet = 0
                    if speaking and not barged and loud_blocks >= INTERRUPT_BLOCKS:
                        barged = True
                        ui.log("system", "✋ interrupted the engineer")
                        self.on_barge_in()
                elif buf:
                    buf.append(block)
                    quiet += 1
                else:
                    peak = max(peak, rms)
                    idle += 1
                    if idle >= HINT_EVERY_BLOCKS:
                        mic(f"listening (loudest sound in the last 10 s: {peak:.4f}, speech needs > {threshold:.4f})")
                        idle, peak = 0, 0.0
                if buf and (quiet >= END_SILENCE_BLOCKS or len(buf) >= MAX_BLOCKS):
                    audio = np.concatenate(buf)
                    enough = loud_blocks >= MIN_SPEECH_BLOCKS
                    buf, loud_blocks, quiet, barged = [], 0, 0, False
                    if not enough:
                        continue  # a click or bump, not speech
                    if len(audio) >= MAX_BLOCKS * BLOCK:
                        ui.log("warn", "20 s of continuous sound: is there constant background noise?")
                    mic("transcribing")
                    t = asyncio.create_task(self._transcribe_into(audio, out))
                    tasks.add(t)
                    t.add_done_callback(tasks.discard)

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
