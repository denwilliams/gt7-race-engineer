"""Configuration: config.toml (optional) with env var overrides."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    """An OpenAI-compatible endpoint. Works with LM Studio, Ollama, etc."""

    base_url: str = "http://localhost:1234/v1"
    model: str = "local-model"
    headers: dict[str, str] = field(default_factory=dict)
    timeout: float = 30.0
    reasoning_effort: str = ""  # e.g. "none" to switch off thinking on reasoning models (Ollama/OpenAI-style)
    kind: str = "chat"  # decision model only: "chat" (one-word prompt) or "jev" (Decisions API)
    api_key_env: str = ""  # name of an env var holding a bearer token, e.g. OPENROUTER_API_KEY

    def auth_headers(self) -> dict[str, str]:
        key = os.environ.get(self.api_key_env, "") if self.api_key_env else ""
        return {**({"Authorization": f"Bearer {key}"} if key else {}), **self.headers}


@dataclass
class TTSConfig:
    """Text-to-speech engine. Empty fields take the engine's default."""

    engine: str = "say"  # "say" (macOS, robotic) | "qwen3" | "kokoro"
    voice: str = ""  # say: "Samantha"; kokoro: "am_adam"; qwen3: "Ryan"
    model: str = ""  # Hugging Face repo for qwen3/kokoro
    speed: float = 1.0  # say: words/min scale; kokoro: speed multiplier; qwen3: ignored
    instruct: str = ""  # qwen3 only: style instruction


@dataclass
class Config:
    ps_ip: str = ""
    engineer: ModelConfig = field(default_factory=ModelConfig)
    lap_reviewer: ModelConfig = field(default_factory=ModelConfig)
    decision: ModelConfig | None = None  # None -> rule-based thresholds only
    stt_model: str = "mlx-community/whisper-small-mlx"
    tts: TTSConfig = field(default_factory=TTSConfig)
    min_seconds_between_calls: float = 8.0


def _model(d: dict | None, default: ModelConfig | None) -> ModelConfig | None:
    if d is None:
        return default
    return ModelConfig(**{**(default.__dict__ if default else {}), **d})


def load(path: str | Path = "config.toml") -> Config:
    raw: dict = {}
    p = Path(path)
    if p.exists():
        raw = tomllib.loads(p.read_text())
    base = ModelConfig()
    cfg = Config(
        ps_ip=raw.get("ps_ip", ""),
        engineer=_model(raw.get("engineer"), base),
        lap_reviewer=_model(raw.get("lap_reviewer"), _model(raw.get("engineer"), base)),
        decision=_model(raw.get("decision"), None),
        stt_model=raw.get("stt_model", Config.stt_model),
        tts=TTSConfig(**raw.get("tts", {})),
    )
    cfg.ps_ip = os.environ.get("GT7_PS_IP", cfg.ps_ip)
    return cfg
