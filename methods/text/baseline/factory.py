"""Construct E1 methods without mixing their memory policies."""

from __future__ import annotations

from typing import Any, Callable

from .base import MemoryMethod
from .full_history import BudgetedFullHistory, FullHistory
from .irmem import IRMem
from .qwen_backend import QwenBackend
from .rolling_summary import RollingSummary
from .wcm import WorldCodeMemory


METHODS = ("full_history", "budgeted_full_history", "rolling_summary", "irmem", "wcm")


def make_factory(
    *, method: str, model_path: str | None, device: str, representation: str = "structured", history_budget_events: int = 4,
    summary_chars: int = 1600, irmem_checkpoint: str | None = None,
) -> tuple[Callable[[], MemoryMethod], dict[str, Any]]:
    if method not in METHODS:
        raise ValueError(f"unknown E1 method: {method}")
    if method == "irmem":
        if not irmem_checkpoint:
            raise ValueError("IRMem requires --irmem-checkpoint")
        controller = IRMem(
            checkpoint_path=irmem_checkpoint,
            device=device,
            representation=representation,
            model_path_override=model_path,
        )
        return lambda: controller, {
            "irmem_checkpoint": irmem_checkpoint,
            "model_path_supplied": model_path is not None,
            "device": device,
            "representation": representation,
        }
    if not model_path:
        raise ValueError(f"{method} requires --model-path")
    backend = QwenBackend(model_path=model_path, device=device)
    if method == "full_history":
        return lambda: FullHistory(backend, representation=representation), {"model_path": backend.model_path, "device": device, "decoding": "greedy", "representation": representation}
    if method == "budgeted_full_history":
        return lambda: BudgetedFullHistory(backend, max_events=history_budget_events, representation=representation), {"model_path": backend.model_path, "device": device, "decoding": "greedy", "history_budget_events": history_budget_events, "representation": representation}
    if method == "rolling_summary":
        return lambda: RollingSummary(backend, representation=representation, max_chars=summary_chars), {"model_path": backend.model_path, "device": device, "decoding": "greedy", "summary_chars": summary_chars, "representation": representation}
    return lambda: WorldCodeMemory(backend, representation=representation), {"model_path": backend.model_path, "device": device, "decoding": "greedy", "representation": representation}
