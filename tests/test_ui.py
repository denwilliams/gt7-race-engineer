import asyncio
import io
import time

from rich.console import Console

from gt7_engineer import ui
from gt7_engineer.state import StateStore
from gt7_engineer.telemetry import simulator
from gt7_engineer.telemetry.stats import IngestStats


def render(r: ui.RichUI) -> str:
    buf = io.StringIO()
    Console(file=buf, width=130, height=40, force_terminal=False, color_system=None).print(r.render())
    return buf.getvalue()


def make(frames=0, **kw):
    console = Console(file=io.StringIO(), width=130, height=40)
    r = ui.RichUI(console=console)
    store, stats = StateStore(), IngestStats()

    async def go():
        n = 0
        async for f in simulator.frames(realtime=False, lap_seconds=3, total_laps=4, **kw):
            stats.frame()
            store.update(f)
            n += 1
            if frames and n >= frames:
                break

    asyncio.run(go())
    r.attach(store=store, stats=stats, info={"ps_ip": "172.16.3.1", "LLM": "gemma4 @ box:11434"})
    return r, store, stats


def test_waiting_before_any_packet():
    r = ui.RichUI(console=Console(file=io.StringIO(), width=130, height=40))
    r.attach(stats=IngestStats(), info={"ps_ip": "172.16.3.1"})
    badge, _, detail = r.telemetry_status()
    assert badge == "WAITING" and "172.16.3.1" in detail
    assert "WAITING" in render(r)


def test_bad_data_when_packets_fail_decryption():
    r = ui.RichUI(console=Console(file=io.StringIO(), width=130, height=40))
    st = IngestStats(received=50, rejected=50)
    r.attach(stats=st)
    badge, _, detail = r.telemetry_status()
    assert badge == "BAD DATA" and "decryption" in detail


def test_live_then_stale():
    r, store, stats = make(frames=300)
    badge, _, detail = r.telemetry_status(time.monotonic())
    assert badge == "LIVE" and "Hz" in detail
    badge, _, detail = r.telemetry_status(time.monotonic() + 5)
    assert badge == "STALE" and "no data" in detail


def test_dashboard_shows_car_state_and_conversation():
    r, store, stats = make(frames=400, start_fuel=40.0)
    r.log("engineer", "Fuel is critical, box this lap.")
    r.log("you", "how is my fuel")
    r.log("lap", "Lap 1: 0:03.000, fuel used 7.0 L")
    r.set_mic("hearing you")
    r.set_engineer("speaking")
    out = render(r)
    for needle in ("LIVE", "Speed", "Fuel", "Tyres", "hearing you", "speaking",
                   "ENGINEER" if False else "Engineer", "Fuel is critical", "how is my fuel", "Lap 1", "gemma4"):
        assert needle in out, f"{needle!r} missing from dashboard:\n{out}"


def test_plain_ui_prints_lines(capsys):
    p = ui.PlainUI()
    p.log("engineer", "Box this lap.")
    p.log("you", "ok")
    p.set_mic("hearing you")
    out = capsys.readouterr().out
    assert "ENGINEER: Box this lap." in out and 'You said: "ok"' in out and "hearing you" in out
