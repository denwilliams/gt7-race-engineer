import asyncio
import json
import os
import stat
import sys

import pytest

from gt7_engineer.config import ModelConfig, load
from gt7_engineer.llm import ChatClient, PiClient, make_chat

FAKE = r'''#!{python}
import json, os, sys
log = os.environ["FAKE_PI_LOG"]
with open(log, "a") as f:
    f.write(json.dumps({{"argv": sys.argv[1:], "pid": os.getpid()}}) + "\n")
last = None
def out(o): sys.stdout.write(json.dumps(o) + "\n"); sys.stdout.flush()
for line in sys.stdin:
    c = json.loads(line)
    t, rid = c["type"], c.get("id")
    if t == "new_session":
        last = None
        out({{"id": rid, "type": "response", "command": "new_session", "success": True, "data": {{"cancelled": False}}}})
    elif t == "prompt":
        if os.environ.get("FAKE_PI_MODE") == "die":
            sys.exit(3)
        out({{"id": rid, "type": "response", "command": "prompt", "success": True, "data": {{"disposition": "started"}}}})
        out({{"type": "agent_start"}})
        last = "echo: " + c["message"]
        out({{"type": "message_update", "assistantMessageEvent": {{"type": "text_delta", "contentIndex": 0, "delta": last}}}})
        out({{"type": "agent_end", "messages": [], "willRetry": False}})
        out({{"type": "agent_settled"}})
    elif t == "get_last_assistant_text":
        out({{"id": rid, "type": "response", "command": "get_last_assistant_text", "success": True, "data": {{"text": last}}}})
'''


@pytest.fixture
def fake_pi(tmp_path, monkeypatch):
    exe = tmp_path / "fakepi"
    exe.write_text(FAKE.format(python=sys.executable))
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("FAKE_PI_LOG", str(log))
    return str(exe), log


def launches(log):
    return [json.loads(l) for l in log.read_text().splitlines()]


def cfg(**kw):
    return ModelConfig(kind="pi", provider="openai-codex", model="gpt-x", reasoning_effort="none", timeout=10, **kw)


async def test_chat_roundtrip_flags_and_process_reuse(fake_pi):
    exe, log = fake_pi
    c = PiClient(cfg(), command=exe)
    a = await c.chat([{"role": "system", "content": "Be brief."}, {"role": "user", "content": "how's my fuel?"}])
    b = await c.chat([{"role": "user", "content": "and tyres?"}])
    await c.aclose()
    assert a.startswith("echo: Instructions:\nBe brief.") and "how's my fuel?" in a
    assert b == "echo: and tyres?"
    ls = launches(log)
    assert len(ls) == 1  # one long-lived process served both calls
    argv = ls[0]["argv"]
    assert argv[:2] == ["--mode", "rpc"]
    for flag in ("--no-session", "--no-tools", "--no-extensions", "--no-skills", "--no-mcp"):
        assert flag in argv
    assert argv[argv.index("--provider") + 1] == "openai-codex"
    assert argv[argv.index("--model") + 1] == "gpt-x"
    assert argv[argv.index("--thinking") + 1] == "off"  # reasoning_effort "none" -> pi "off"


async def test_history_is_rendered_into_the_prompt(fake_pi):
    exe, _ = fake_pi
    c = PiClient(cfg(), command=exe)
    out = await c.chat([
        {"role": "system", "content": "State: lap 3"},
        {"role": "user", "content": "how's fuel?"},
        {"role": "assistant", "content": "Fine."},
        {"role": "user", "content": "and now?"},
    ])
    await c.aclose()
    assert "State: lap 3" in out and "user: how's fuel?" in out and "assistant: Fine." in out
    assert out.rstrip().endswith("and now?")


async def test_crash_raises_then_next_call_restarts(fake_pi, monkeypatch):
    exe, log = fake_pi
    c = PiClient(cfg(), command=exe)
    monkeypatch.setenv("FAKE_PI_MODE", "die")
    with pytest.raises(RuntimeError, match="exited unexpectedly"):
        await c.chat([{"role": "user", "content": "hi"}])
    monkeypatch.delenv("FAKE_PI_MODE")
    assert await c.chat([{"role": "user", "content": "again"}]) == "echo: again"
    await c.aclose()
    assert len(launches(log)) == 2  # restarted


async def test_missing_pi_gives_clear_error():
    c = PiClient(cfg(), command="definitely-not-installed-pi")
    with pytest.raises(RuntimeError, match="not found on PATH"):
        await c.chat([{"role": "user", "content": "hi"}])


def test_config_selects_pi_and_reviewer_inherits(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text('[engineer]\nkind = "pi"\nprovider = "openai-codex"\nmodel = "gpt-x"\nreasoning_effort = "none"\n[lap_reviewer]\nmodel = "gpt-y"\n')
    c = load(p)
    assert isinstance(make_chat(c.engineer), PiClient)
    assert c.lap_reviewer.kind == "pi" and c.lap_reviewer.provider == "openai-codex" and c.lap_reviewer.model == "gpt-y"
    assert isinstance(make_chat(ModelConfig()), ChatClient)
