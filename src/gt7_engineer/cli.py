from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from . import app, config
from .audio.tts import PrintTTS, make_tts
from .state import StateStore
from .telemetry import ingest, simulator


def _source(args, cfg):
    if args.simulate:
        return simulator.frames(
            lap_seconds=args.lap_seconds, total_laps=args.laps, speedup=args.speedup,
            realtime=not args.fast,
        )
    if not cfg.ps_ip:
        sys.exit("Set the PlayStation IP: --ps-ip 192.168.x.x, GT7_PS_IP, or ps_ip in config.toml (or use --simulate)")
    return ingest.frames(cfg.ps_ip)


async def _run(args) -> None:
    cfg = config.load(args.config)
    if args.ps_ip:
        cfg.ps_ip = args.ps_ip
    speaking = asyncio.Event()
    if args.text:
        from .audio.stt import stdin_utterances

        speak, utterances = PrintTTS().speak, stdin_utterances()
    elif args.simulate and args.no_audio:
        speak, utterances = PrintTTS().speak, None
    else:
        from .audio.stt import WhisperListener

        tts = make_tts(cfg.tts)
        listener = WhisperListener(cfg.stt_model, speaking)
        print(
            f"Loading speech recognition ({cfg.stt_model}) and voice ({cfg.tts.engine}); "
            "the first run downloads the models...",
            flush=True,
        )
        await asyncio.gather(listener.warmup(), tts.warmup())  # Whisper starts loading immediately, in parallel

        async def speak(text: str) -> None:
            print(f"\n🎙  ENGINEER: {text}\n", flush=True)  # exact text handed to the TTS engine
            await tts.speak(text)

        utterances = listener.utterances()
    await app.run(cfg, _source(args, cfg), speak, utterances, use_llm=not args.no_llm, speaking=speaking)


async def _watch(args) -> None:
    cfg = config.load(args.config)
    if args.ps_ip:
        cfg.ps_ip = args.ps_ip
    store = StateStore()
    n = 0
    async for f in _source(args, cfg):
        ev = store.update(f)
        n += 1
        if ev:
            print(f"--- lap {ev.record.lap} done: {ev.record.time_ms} ms, fuel used {ev.record.fuel_used:.2f} L")
        if n % 30 == 0:
            print(store.summary(), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(prog="gt7-engineer", description="Local AI race engineer for GT7")
    p.add_argument("command", choices=["run", "watch"], help="run = full engineer, watch = telemetry readout only")
    p.add_argument("--config", default="config.toml")
    p.add_argument("--ps-ip", help="PlayStation IP address")
    p.add_argument("--simulate", action="store_true", help="use a fake race instead of a PlayStation")
    p.add_argument("--laps", type=int, default=5, help="simulated race length")
    p.add_argument("--lap-seconds", type=float, default=30.0)
    p.add_argument("--speedup", type=float, default=1.0)
    p.add_argument("--fast", action="store_true", help="simulate without real-time pacing")
    p.add_argument("--text", action="store_true", help="type to the engineer and print replies (no mic/speaker)")
    p.add_argument("--no-audio", action="store_true", help="print only (simulate mode)")
    p.add_argument("--no-llm", action="store_true", help="no language model: rule-based calls only")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")
    try:
        asyncio.run(_run(args) if args.command == "run" else _watch(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
