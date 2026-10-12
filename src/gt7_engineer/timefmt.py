"""Lap-time formatting shared by the state summary, the UI and the CLI."""

from __future__ import annotations


def fmt_lap(ms: int | None) -> str:
    """92400 -> '1:32.400'; missing/invalid -> '-'."""
    if not ms or ms <= 0:
        return "-"
    m, s = divmod(ms / 1000, 60)
    return f"{int(m)}:{s:06.3f}"
