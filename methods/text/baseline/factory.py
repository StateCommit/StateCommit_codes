"""Construct E1 methods without mixing their memory policies."""

from __future__ import annotations

from typing import Any, Callable

from .base import MemoryMethod
from .qwen_backend import QwenBackend
from .wcm import WorldCodeMemory


METHODS = ("wcm",)


def make_factory(
    *, method: str, model_path: str | None, device: str, representation: str = "structured",
) -> tuple[Callable[[], MemoryMethod], dict[str, Any]]:
    if method not in METHODS:
        raise ValueError(f"unknown E1 method: {method}")
    if not model_path:
        raise ValueError(f"{method} requires --model-path")
    backend = QwenBackend(model_path=model_path, device=device)
    return lambda: WorldCodeMemory(backend, representation=representation), {"model_path": backend.model_path, "device": device, "decoding": "greedy", "representation": representation}
