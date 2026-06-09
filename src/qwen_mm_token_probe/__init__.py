"""Utilities for probing token probabilities in multimodal Qwen-style models."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .probe import ProbeResult, run_probe

__all__ = ["ProbeResult", "run_probe"]


def __getattr__(name: str):
    if name in __all__:
        from .probe import ProbeResult, run_probe

        return {"ProbeResult": ProbeResult, "run_probe": run_probe}[name]
    raise AttributeError(name)
