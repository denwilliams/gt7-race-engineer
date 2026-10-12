"""Text-to-speech. Default: the macOS `say` command (local, zero setup).

Swap in a neural voice (Kokoro, Piper, ...) by implementing `speak(text)`.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Protocol

log = logging.getLogger(__name__)


class TTS(Protocol):
    async def speak(self, text: str) -> None: ...


class SayTTS:
    def __init__(self, voice: str = "Samantha", rate: int = 210):
        self.voice, self.rate = voice, rate

    async def speak(self, text: str) -> None:
        proc = await asyncio.create_subprocess_exec("say", "-v", self.voice, "-r", str(self.rate), text)
        await proc.wait()


class PrintTTS:
    """Text-only output for development without audio."""

    async def speak(self, text: str) -> None:
        print(f"\n🎙  ENGINEER: {text}\n", flush=True)
