"""Terminal UI. One module-level instance (`ui.ui`) that the rest of the app talks to.

PlainUI prints lines (pipes, --text mode, --plain). RichUI is a live dashboard:
telemetry health and car state, mic/engineer status, and the conversation.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass

# kind -> (icon, rich style, plain label)
KINDS = {
    "engineer": ("🎙", "bold cyan", "ENGINEER"),
    "you": ("📝", "bold green", "You said"),
    "analyst": ("📊", "yellow", "analyst"),
    "lap": ("🏁", "magenta", "lap"),
    "system": ("•", "white", ""),
    "warn": ("⚠", "bold red", "warning"),
}


def fmt_lap(ms: int | None) -> str:
    if not ms or ms <= 0:
        return "-"
    m, s = divmod(ms / 1000, 60)
    return f"{int(m)}:{s:06.3f}"


class PlainUI:
    interactive = False

    def attach(self, store=None, stats=None, info: dict | None = None) -> None: ...
    def start(self) -> None: ...
    def close(self) -> None: ...
    def set_mic(self, status: str, show: bool = True) -> None:
        if show:
            print(f"🎤 {status}", flush=True)

    def set_engineer(self, status: str) -> None: ...

    def log(self, kind: str, text: str) -> None:
        icon, _, label = KINDS.get(kind, KINDS["system"])
        if kind == "engineer":
            print(f"\n{icon}  {label}: {text}\n", flush=True)
        elif kind == "you":
            print(f'{icon} {label}: "{text}"', flush=True)
        else:
            print(f"{icon} {text}", flush=True)


@dataclass
class _Entry:
    t: float
    kind: str
    text: str


class RichUI:
    interactive = True

    def __init__(self, console=None, max_entries: int = 300):
        from rich.console import Console

        self.console = console or Console()
        self.entries: deque[_Entry] = deque(maxlen=max_entries)
        self.mic = "starting"
        self.engineer = "idle"
        self.store = None
        self.stats = None
        self.info: dict = {}
        self.started = time.monotonic()
        self._live = None

    # ---- feed ---------------------------------------------------------
    def attach(self, store=None, stats=None, info: dict | None = None) -> None:
        self.store = store if store is not None else self.store
        self.stats = stats if stats is not None else self.stats
        if info:
            self.info.update(info)

    def set_mic(self, status: str, show: bool = True) -> None:
        self.mic = status

    def set_engineer(self, status: str) -> None:
        self.engineer = status

    def log(self, kind: str, text: str) -> None:
        self.entries.append(_Entry(time.monotonic(), kind, text))

    # ---- lifecycle ----------------------------------------------------
    def start(self) -> None:
        from rich.live import Live

        self._live = Live(get_renderable=self.render, console=self.console, screen=True, refresh_per_second=8)
        self._live.start()

    def close(self) -> None:
        if self._live is not None:
            try:
                self._live.stop()
            except Exception:
                pass
            self._live = None

    # ---- rendering ----------------------------------------------------
    def telemetry_status(self, now: float | None = None) -> tuple[str, str, str]:
        """(badge, rich colour, detail) describing the telemetry link."""
        st = self.stats
        if st is None:
            return "NO LINK", "dim", ""
        now = now if now is not None else time.monotonic()
        ip = self.info.get("ps_ip", "")
        if st.decoded == 0:
            if st.rejected > 0:
                return "BAD DATA", "bold red", f"{st.rejected} packets arrived but failed decryption (wrong key/format?)"
            where = f" from {ip}" if ip else ""
            return "WAITING", "bold yellow", f"no packets{where} yet. Is a race running? Same network?"
        age = st.age(now)
        if age is not None and age > 2.0:
            return "STALE", "bold red", f"no data for {age:.0f}s"
        car = self.store.car if self.store else None
        if car is not None and not car.on_track:
            return "PAUSED", "bold yellow", f"{st.hz(now):.0f} Hz, car not on track (paused/menu/loading)"
        return "LIVE", "bold green", f"{st.hz(now):.0f} Hz"

    def _telemetry_panel(self, now: float):
        from rich.panel import Panel
        from rich.table import Table

        badge, colour, detail = self.telemetry_status(now)
        t = Table.grid(padding=(0, 2))
        t.add_column(style="dim", no_wrap=True)
        t.add_column()
        t.add_row("Telemetry", f"[{colour}]● {badge}[/]  {detail}")
        s = self.stats
        if s is not None:
            t.add_row("Packets", f"{s.received} received · {s.decoded} decoded · {s.rejected} rejected")
        car = self.store.car if self.store else None
        if car is not None:
            sm = self.store.summary()
            laps = f"{car.lap} / {car.total_laps}" if car.total_laps else f"{car.lap}"
            t.add_row("Lap", laps)
            grid = f"P{sm['starting_position']}" + (f" of {sm['cars_in_race']}" if sm.get("cars_in_race") else "") if sm.get("starting_position") else "?"
            now_pos = sm.get("current_position", "")
            now_pos = now_pos.split("driver last said: ")[1] if "driver last said" in now_pos else "n/a (not in GT7 telemetry)"
            t.add_row("Position", f"start {grid}   now {now_pos}")
            t.add_row("Speed", f"{sm['speed_kph']} km/h   gear {car.gear}   {car.rpm:.0f} rpm")
            t.add_row("Throttle", _bar(car.throttle, "green") + "   Brake " + _bar(car.brake, "red"))
            fuel = f"{sm['fuel_l']} L"
            if sm.get("fuel_pct") is not None:
                fuel += f" ({sm['fuel_pct']}%)"
            t.add_row("Fuel", fuel)
            if "laps_of_fuel" in sm:
                m = sm.get("fuel_margin_laps")
                extra = f"   margin {m:+.1f} laps" if m is not None else ""
                col = "red" if (m is not None and m < 0) else "white"
                t.add_row("", f"[{col}]{sm['laps_of_fuel']} laps of fuel{extra}[/]")
            if sm.get("fuel_per_lap_l"):
                t.add_row("Burn", f"{sm['fuel_per_lap_l']} L/lap   {sm.get('fuel_burn_trend', '')}")
            if sm.get("fuel_at_lap_start_l"):
                starts = list(sm["fuel_at_lap_start_l"].items())[-6:]
                t.add_row("Fuel @ lap start", "  ".join(f"L{lap} {v}" for lap, v in starts))
            t.add_row("Last / best", f"{fmt_lap(car.last_lap_ms)} / {fmt_lap(car.best_lap_ms)}")
            fl, fr, rl, rr = (round(x) for x in car.tyre_temp)
            t.add_row("Tyres °C", f"FL {fl}  FR {fr}\n RL {rl}  RR {rr}")
            if "driver_reported_tyre_wear" in sm:
                t.add_row("Tyre wear", f"(you said) {sm['driver_reported_tyre_wear']}")
        return Panel(t, title="Car", border_style=colour if badge != "LIVE" else "green")

    def _status_panel(self):
        from rich.panel import Panel
        from rich.table import Table

        t = Table.grid(padding=(0, 2))
        t.add_column(style="dim", no_wrap=True)
        t.add_column()
        mic_col = {"hearing you": "green", "transcribing": "cyan", "muted (engineer talking)": "yellow"}.get(self.mic, "white")
        eng_col = {"speaking": "cyan", "thinking": "yellow"}.get(self.engineer, "white")
        t.add_row("Mic", f"[{mic_col}]{self.mic}[/]")
        t.add_row("Engineer", f"[{eng_col}]{self.engineer}[/]")
        for k, v in self.info.items():
            if k != "ps_ip":
                t.add_row(k, str(v))
        t.add_row("Up", f"{int(time.monotonic() - self.started)}s   [dim]Ctrl-C to quit[/]")
        return Panel(t, title="Engineer", border_style="cyan")

    def _log_panel(self, rows: int):
        from rich.panel import Panel
        from rich.text import Text

        out = Text(no_wrap=False, overflow="fold")
        shown = list(self.entries)[-max(rows, 1):]
        for i, e in enumerate(shown):
            icon, style, label = KINDS.get(e.kind, KINDS["system"])
            stamp = time.strftime("%H:%M:%S", time.localtime(time.time() - (time.monotonic() - e.t)))
            out.append(f"{stamp} ", style="dim")
            out.append(f"{icon} ", style=style)
            if e.kind in ("engineer", "you", "analyst"):
                out.append(f"{label}: ", style=style)
            out.append(e.text, style=style if e.kind in ("engineer", "warn") else "")
            if i < len(shown) - 1:
                out.append("\n")
        return Panel(out, title="Conversation", border_style="dim")

    def render(self):
        from rich.layout import Layout

        now = time.monotonic()
        height = self.console.size.height
        top = 16
        root = Layout()
        root.split_column(Layout(name="top", size=top), Layout(name="log"))
        root["top"].split_row(Layout(self._telemetry_panel(now), ratio=3), Layout(self._status_panel(), ratio=2))
        root["log"].update(self._log_panel(height - top - 3))
        return root


def _bar(v: float, colour: str, width: int = 10) -> str:
    n = round(max(0.0, min(1.0, v)) * width)
    return f"[{colour}]{'█' * n}[/][dim]{'░' * (width - n)}[/]"


class UIHandler(logging.Handler):
    """Route log records (warnings, errors) into the UI so they don't tear the display."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if record.levelno >= logging.WARNING:
                ui.log("warn", record.getMessage())
        except Exception:
            pass


ui: PlainUI | RichUI = PlainUI()


def use(new: PlainUI | RichUI) -> None:
    global ui
    ui = new


# Forwarders: callers do `from . import ui; ui.log(...)` and always reach the active UI.
def log(kind: str, text: str) -> None:
    ui.log(kind, text)


def set_mic(status: str, show: bool = True) -> None:
    ui.set_mic(status, show)


def set_engineer(status: str) -> None:
    ui.set_engineer(status)


def attach(store=None, stats=None, info: dict | None = None) -> None:
    ui.attach(store, stats, info)
