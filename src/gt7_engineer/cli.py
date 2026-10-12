from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
from urllib.parse import urlparse

from . import app, config, ui
from .audio.tts import PrintTTS, make_tts
from .state import NewRace, StateStore
from .telemetry import ingest, simulator
from .telemetry.stats import IngestStats


def llm_label(m: config.ModelConfig) -> str:
    """How the dashboard names a model: Pi has no URL, so don't show the (unused) default base_url."""
    if m.kind == "pi":
        return f"{m.model} via Pi ({m.provider})"
    return f"{m.model} @ {urlparse(m.base_url).netloc}"


def _source(args, cfg, stats: IngestStats):
    if args.simulate:
        return simulator.frames(
            lap_seconds=args.lap_seconds, total_laps=args.laps, speedup=args.speedup,
            realtime=not args.fast,
        )
    if not cfg.ps_ip:
        sys.exit("Set the PlayStation IP: --ps-ip 192.168.x.x, GT7_PS_IP, or ps_ip in config.toml (or use --simulate)")
    return ingest.frames(cfg.ps_ip, stats=stats)


def _start_keys(reset_request: asyncio.Event) -> None:
    """`r` asks for a race-state reset. Only on the live dashboard (a real terminal)."""
    if ui.ui.interactive:
        keys = ui.KeyListener({"r": reset_request.set})
        if keys.start(asyncio.get_running_loop()):
            ui.ui.keys = keys


def _setup_ui(args) -> None:
    """Live dashboard on a real terminal; plain lines for pipes, --text, --plain."""
    if sys.stdout.isatty() and not args.plain and not args.text:
        ui.use(ui.RichUI())
        root = logging.getLogger()
        root.handlers = [ui.UIHandler()]  # warnings/errors go into the dashboard, not over it
        ui.ui.start()


async def _run(args) -> None:
    cfg = config.load(args.config)
    if args.ps_ip:
        cfg.ps_ip = args.ps_ip
    stats = IngestStats()
    info = {
        "ps_ip": "simulator" if args.simulate else cfg.ps_ip,
        "Source": "simulator" if args.simulate else f"PlayStation {cfg.ps_ip}",
    }
    if not args.no_llm:
        info["LLM"] = llm_label(cfg.engineer)
    speaking = asyncio.Event()
    stop_speaking = interrupter = None
    if args.text:
        from .audio.stt import stdin_utterances

        speak, utterances = PrintTTS().speak, stdin_utterances()
    elif args.simulate and args.no_audio:
        speak, utterances = PrintTTS().speak, None
    else:
        from .audio.stt import WhisperListener

        tts = make_tts(cfg.tts)
        interrupter = app.Interrupter()
        listener = WhisperListener(cfg.stt_model, speaking, interrupter.fire if cfg.audio.barge_in else None)
        stop_speaking = tts.stop
        info["Voice"] = f"{cfg.tts.engine} / {getattr(tts, 'voice', '') or 'default'}"
        info["Interrupt"] = "mic open (headphones)" if cfg.audio.barge_in else "mic muted while it talks"
        ui.attach(stats=stats, info=info)
        ui.set_mic("loading speech recognition")
        ui.log("system", f"Loading speech recognition ({cfg.stt_model}) and voice ({cfg.tts.engine}); the first run downloads models...")
        await asyncio.gather(listener.warmup(), tts.warmup())  # Whisper starts loading immediately, in parallel
        speak, utterances = tts.speak, listener.utterances()
    ui.attach(stats=stats, info=info)
    reset_request = asyncio.Event()
    _start_keys(reset_request)
    await app.run(
        cfg, _source(args, cfg, stats), speak, utterances, use_llm=not args.no_llm, speaking=speaking,
        stop_speaking=stop_speaking, interrupter=interrupter, stats=stats, reset_request=reset_request,
    )


async def _watch(args) -> None:
    cfg = config.load(args.config)
    if args.ps_ip:
        cfg.ps_ip = args.ps_ip
    stats, store = IngestStats(), StateStore()
    ui.attach(store=store, stats=stats, info={"ps_ip": "simulator" if args.simulate else cfg.ps_ip})
    ui.set_mic("n/a (watch mode)", show=False)
    ui.log("system", "Watching telemetry only (no engineer, no mic).")
    n = 0
    reset_request = asyncio.Event()
    _start_keys(reset_request)
    async for f in _source(args, cfg, stats):
        stats.frame()
        if reset_request.is_set():
            reset_request.clear()
            store.reset()
            ui.log("system", "🔄 Race state reset (you pressed r)")
        ev = store.update(f)
        n += 1
        if isinstance(ev, NewRace):
            ui.log("system", f"🔄 New race detected ({ev.reason}): state reset")
            ev = None
        if ev:
            ui.log("lap", f"Lap {ev.record.lap}: {ui.fmt_lap(ev.record.time_ms)}, fuel used {ev.record.fuel_used:.2f} L")
        if not ui.ui.interactive and n % 30 == 0:
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
    p.add_argument("--plain", action="store_true", help="plain text lines instead of the live dashboard")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")
    if not args.verbose:
        for noisy in ("httpx", "httpcore", "huggingface_hub"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
    _setup_ui(args)

    def stop(*_) -> None:
        # Mic, TTS and Whisper run in worker threads that block for a while; a normal
        # shutdown waits for them (hence needing Ctrl-C twice). Nothing needs flushing.
        ui.ui.close()  # leave the dashboard's alternate screen before exiting
        print("\nStopped.", flush=True)
        os._exit(0)

    signal.signal(signal.SIGINT, stop)
    try:
        asyncio.run(_run(args) if args.command == "run" else _watch(args))
    except KeyboardInterrupt:
        pass
    finally:
        ui.ui.close()


if __name__ == "__main__":
    main()
