"""OpenAI-compatible chat client and the decision-model abstraction."""

from __future__ import annotations

import json
import logging
import re
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
        r = await self._http.post(
            "/chat/completions",
            json={
                "model": self.cfg.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()

    async def aclose(self) -> None:
        await self._http.aclose()


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
