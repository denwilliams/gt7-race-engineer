"""OpenAI-compatible chat client and the decision-model abstraction."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
from typing import Protocol

import httpx

from .config import ModelConfig

log = logging.getLogger(__name__)


class ChatClient:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self._http = httpx.AsyncClient(
            base_url=cfg.base_url.rstrip("/"), headers=cfg.auth_headers(), timeout=cfg.timeout
        )

    async def chat(self, messages: list[dict], max_tokens: int = 200, temperature: float = 0.3) -> str:
        body = {
            "model": self.cfg.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if self.cfg.reasoning_effort:
            body["reasoning_effort"] = self.cfg.reasoning_effort
        r = await self._http.post("/chat/completions", json=body)
        r.raise_for_status()
        choice = r.json()["choices"][0]
        text = (choice["message"].get("content") or "").strip()
        if not text and choice.get("finish_reason") == "length":
            log.warning(
                "%s returned no text: it ran out of tokens, probably thinking. "
                'Set reasoning_effort = "none" for this model in config.toml.',
                self.cfg.model,
            )
        return text

    async def aclose(self) -> None:
        await self._http.aclose()


class PiClient:
    """Chat through the Pi coding-agent harness in RPC mode (https://pi.dev).

    Uses whatever Pi is logged in with, e.g. a ChatGPT Plus/Pro (Codex) subscription,
    so no API key is needed. One long-lived `pi --mode rpc` child process serves all
    calls; every call starts a fresh Pi session so there is no hidden history (this app
    sends its own context). Tools, extensions, skills and MCP are disabled: it is used
    as a plain text model. Implements the same `chat()` as ChatClient.
    NOTE: this sends race data to whichever cloud provider Pi is logged in to.
    """

    BASE_SYSTEM = (
        "You are a text model embedded in a race engineer app. Follow the instructions in each "
        "message exactly and reply with only the requested text."
    )

    def __init__(self, cfg: ModelConfig, command: str = "pi"):
        self.cfg, self.command = cfg, command
        self._proc: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._n = 0

    def _argv(self) -> list[str]:
        argv = [
            self.command, "--mode", "rpc", "--no-session", "--no-tools", "--no-extensions", "--no-skills",
            "--no-mcp", "--no-context-files", "--no-prompt-templates", "--system-prompt", self.BASE_SYSTEM,
        ]
        if self.cfg.provider:
            argv += ["--provider", self.cfg.provider]
        if self.cfg.model:
            argv += ["--model", self.cfg.model]
        effort = {"none": "off"}.get(self.cfg.reasoning_effort, self.cfg.reasoning_effort)
        argv += ["--thinking", effort or "off"]
        return argv

    async def _ensure(self) -> asyncio.subprocess.Process:
        if self._proc is None or self._proc.returncode is not None:
            if shutil.which(self.command) is None:
                raise RuntimeError(f"'{self.command}' not found on PATH; install Pi (https://pi.dev) and run /login")
            self._proc = await asyncio.create_subprocess_exec(
                *self._argv(), stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL, limit=1 << 24,
            )
        return self._proc

    async def _send(self, proc, cmd: dict) -> str:
        self._n += 1
        rid = f"gt7-{self._n}"
        proc.stdin.write((json.dumps({**cmd, "id": rid}) + "\n").encode())
        await proc.stdin.drain()
        return rid

    async def _read(self, proc) -> dict:
        line = await proc.stdout.readline()
        if not line:
            raise RuntimeError("pi exited unexpectedly (not logged in? run `pi`, then /login)")
        return json.loads(line)

    async def _response(self, proc, rid: str) -> dict:
        while True:
            msg = await self._read(proc)
            if msg.get("type") == "response" and msg.get("id") == rid:
                if not msg.get("success"):
                    raise RuntimeError(f"pi {msg.get('command')} failed: {msg.get('error')}")
                return msg

    @staticmethod
    def render(messages: list[dict]) -> str:
        """Flatten chat messages into one prompt: instructions first, then the conversation."""
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        turns = [m for m in messages if m["role"] != "system"]
        parts = []
        if system:
            parts.append(f"Instructions:\n{system}")
        if len(turns) == 1:
            parts.append(turns[0]["content"])
        else:
            parts.append("Conversation so far:\n" + "\n".join(f"{m['role']}: {m['content']}" for m in turns[:-1]))
            parts.append(f"Reply to the latest message:\n{turns[-1]['content']}")
        return "\n\n".join(parts)

    async def chat(self, messages: list[dict], max_tokens: int = 200, temperature: float = 0.3) -> str:
        async with self._lock:
            try:
                return await asyncio.wait_for(self._chat(messages), self.cfg.timeout)
            except BaseException:
                await self._kill()  # never leave a wedged process behind; the next call restarts it
                raise

    async def _chat(self, messages: list[dict]) -> str:
        proc = await self._ensure()
        await self._response(proc, await self._send(proc, {"type": "new_session"}))
        await self._response(proc, await self._send(proc, {"type": "prompt", "message": self.render(messages)}))
        while True:  # wait for the run (including any automatic retries) to finish
            if (await self._read(proc)).get("type") == "agent_settled":
                break
        msg = await self._response(proc, await self._send(proc, {"type": "get_last_assistant_text"}))
        return ((msg.get("data") or {}).get("text") or "").strip()

    async def _kill(self) -> None:
        if self._proc is not None and self._proc.returncode is None:
            self._proc.kill()
            await self._proc.wait()
        self._proc = None

    async def aclose(self) -> None:
        await self._kill()


def make_chat(cfg: ModelConfig) -> "ChatClient | PiClient":
    return PiClient(cfg) if cfg.kind == "pi" else ChatClient(cfg)


class DecisionModel(Protocol):
    async def decide(self, instructions: str, context: dict, choices: list[str]) -> str:
        """Pick exactly one of `choices` given calculated `context`."""


class ThresholdDecision:
    """Fallback when no decision model is configured: choose by a callable."""

    def __init__(self, fn):
        self.fn = fn

    async def decide(self, instructions: str, context: dict, choices: list[str]) -> str:
        return self.fn(context)


class OpenAIDecision:
    """Decision via any OpenAI-compatible chat endpoint, constrained to a choice list.

    Intended for Jev-Style or a small instruct model. Jev-Style ships its own
    server whose API has NOT been verified here; if it is not OpenAI-compatible
    write a small adapter implementing DecisionModel.
    """

    def __init__(self, client: ChatClient, default: str):
        self.client = client
        self.default = default

    async def decide(self, instructions: str, context: dict, choices: list[str]) -> str:
        prompt = (
            f"{instructions}\n\nData:\n{json.dumps(context)}\n\n"
            f"Answer with exactly one of: {', '.join(choices)}. Output only that word."
        )
        try:
            out = await self.client.chat([{"role": "user", "content": prompt}], max_tokens=8, temperature=0.0)
        except Exception as e:  # network/model failure must never stop the race
            log.warning("decision model failed: %s", e)
            return self.default
        for c in choices:
            if re.search(rf"\b{re.escape(c)}\b", out, re.I):
                return c
        return self.default


CHOICE_CRITERIA = {
    "ignore": "Nothing the driver needs to hear right now.",
    "mention": "Worth a short heads-up at the next convenient moment.",
    "urgent": "The driver must be told immediately; it affects the race result or safety.",
}


class JevDecision:
    """Decision via the Jev Decisions API (OpenRouter: POST {base_url}/decisions).

    Sends the analyst's calculated numbers as `state` and one `choice` question;
    uses the returned choice. NOTE: this sends race data to the host in base_url,
    so it is not local. Request/response shape follows OpenRouter's Jev tutorial.
    """

    def __init__(self, cfg: ModelConfig, default: str, http: httpx.AsyncClient | None = None):
        self.cfg, self.default = cfg, default
        self._http = http or httpx.AsyncClient(
            base_url=cfg.base_url.rstrip("/"), headers=cfg.auth_headers(), timeout=cfg.timeout
        )
        self.last_probabilities: dict[str, float] = {}

    async def decide(self, instructions: str, context: dict, choices: list[str]) -> str:
        body = {
            "model": self.cfg.model,
            "state": context,
            "questions": {
                "alert": {
                    "type": "choice",
                    "instructions": instructions,
                    "criteria": {c: CHOICE_CRITERIA.get(c, c) for c in choices},
                }
            },
        }
        try:
            r = await self._http.post("/decisions", json=body)
            r.raise_for_status()
            ans = r.json()["answers"]["alert"]
            self.last_probabilities = ans.get("probabilities", {})
            choice = ans["choice"]
        except Exception as e:  # network/model failure must never stop the race
            log.warning("jev decision failed: %s", e)
            return self.default
        return choice if choice in choices else self.default

    async def aclose(self) -> None:
        await self._http.aclose()
