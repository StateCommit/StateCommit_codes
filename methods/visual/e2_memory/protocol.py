"""Public method-side query bindings and typed output constraints.

This module has no evaluator dependency.  It is shared by training and
inference, but all information it handles is already present in `public/`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from e0.public_protocol import PublicBenchmark, PublicQuery
from e1.tracked_state_protocol import parse_slot_key
from e0_5.learned_grounder import SUBJECT_INDEX, VALUE_INDEX, VALUE_TOKENS, visual_subject_key

from .models import COMPONENT_ORDER


COMPONENT_INDEX = {component: index for index, component in enumerate(COMPONENT_ORDER)}
LEGAL_VALUES = {
    "openness": (VALUE_INDEX["closed"], VALUE_INDEX["open"]),
    "toggle_state": (VALUE_INDEX["off"], VALUE_INDEX["on"]),
    "place": tuple(VALUE_INDEX[name] for name in ("border", "checker", "diagonal", "dots", "solid", "vertical")),
}


def _catalog_entry(*, benchmark: PublicBenchmark, stream_id: str, entity_id: str) -> dict[str, object]:
    try:
        catalog = {str(item.get("entity_id")): item for item in benchmark.streams[stream_id].entity_catalog}
    except KeyError as error:
        raise ValueError(f"unknown public stream: {stream_id}") from error
    item = catalog.get(entity_id)
    if item is None:
        raise ValueError(f"public entity {entity_id} is absent from stream {stream_id}")
    return item


def _validate_public_slot(*, benchmark: PublicBenchmark, stream_id: str, entity_id: str, component: str) -> None:
    entity_type = _catalog_entry(benchmark=benchmark, stream_id=stream_id, entity_id=entity_id).get("entity_type")
    if (component == "place" and entity_type not in {"ball", "key"}) or (
        component == "openness" and entity_type != "door"
    ) or (component == "toggle_state" and entity_type != "switch"):
        raise ValueError(f"public slot is type-incompatible: {entity_id}|{component}")


def slot_subject_index(*, benchmark: PublicBenchmark, stream_id: str, entity_id: str, component: str) -> int:
    """Encode one public ``entity|component`` slot for a memory decoder."""

    _validate_public_slot(benchmark=benchmark, stream_id=stream_id, entity_id=entity_id, component=component)
    return SUBJECT_INDEX[visual_subject_key(_catalog_entry(benchmark=benchmark, stream_id=stream_id, entity_id=entity_id))]


def query_subject_index(*, query: PublicQuery, benchmark: PublicBenchmark) -> int:
    return slot_subject_index(
        benchmark=benchmark, stream_id=query.stream_id, entity_id=query.entity_id, component=query.component
    )


def decode_slot_value(
    *, benchmark: PublicBenchmark, stream_id: str, entity_id: str, component: str, value_index: int
) -> str | dict[str, str]:
    """Compile a component-restricted class through public catalog bindings."""

    _validate_public_slot(benchmark=benchmark, stream_id=stream_id, entity_id=entity_id, component=component)
    if value_index not in LEGAL_VALUES[component]:
        raise ValueError("cannot decode an illegal value class for query component")
    value = VALUE_TOKENS[value_index]
    if component in {"openness", "toggle_state"}:
        return value
    for entry in benchmark.streams[stream_id].entity_catalog:
        if entry.get("entity_type") == "zone" and entry.get("pattern") == value:
            entity_id = entry.get("entity_id")
            if isinstance(entity_id, str):
                return {"kind": "in_zone", "target": entity_id}
    raise ValueError(f"public catalog has no zone binding for visual pattern {value!r}")


def decode_value(*, query: PublicQuery, value_index: int, benchmark: PublicBenchmark) -> str | dict[str, str]:
    return decode_slot_value(
        benchmark=benchmark,
        stream_id=query.stream_id,
        entity_id=query.entity_id,
        component=query.component,
        value_index=value_index,
    )


def slot_value_mask(*, benchmark: PublicBenchmark, stream_id: str, entity_id: str, component: str) -> torch.Tensor:
    """Return the public, stream-specific legal output vocabulary for a query."""

    _validate_public_slot(benchmark=benchmark, stream_id=stream_id, entity_id=entity_id, component=component)
    allowed = torch.zeros(len(VALUE_TOKENS), dtype=torch.bool)
    if component != "place":
        allowed[list(LEGAL_VALUES[component])] = True
        return allowed
    for entry in benchmark.streams[stream_id].entity_catalog:
        pattern = entry.get("pattern") if entry.get("entity_type") == "zone" else None
        if isinstance(pattern, str) and pattern in VALUE_INDEX:
            allowed[VALUE_INDEX[pattern]] = True
    if not bool(allowed.any()):
        raise ValueError(f"public catalog has no valid place patterns for {stream_id}")
    return allowed


def query_value_mask(*, query: PublicQuery, benchmark: PublicBenchmark) -> torch.Tensor:
    return slot_value_mask(
        benchmark=benchmark, stream_id=query.stream_id, entity_id=query.entity_id, component=query.component
    )


def slot_binding(*, benchmark: PublicBenchmark, stream_id: str, slot: str) -> tuple[int, int, torch.Tensor]:
    """Return subject/component/mask from a public scope's slot string."""

    entity_id, component = parse_slot_key(slot)
    return (
        slot_subject_index(benchmark=benchmark, stream_id=stream_id, entity_id=entity_id, component=component),
        COMPONENT_INDEX[component],
        slot_value_mask(benchmark=benchmark, stream_id=stream_id, entity_id=entity_id, component=component),
    )


def restrict_logits(logits: torch.Tensor, components: torch.Tensor, allowed_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Mask invalid component and public-catalog values for every method.

    ``allowed_mask`` is required by real dataset runners and is derived only
    from the public entity catalog.  The optional default keeps small unit
    tests concise while still enforcing the global component type.
    """

    if logits.ndim != 2 or components.ndim != 1 or logits.shape[0] != components.shape[0]:
        raise ValueError("logit/component shapes are malformed")
    allowed = torch.zeros_like(logits, dtype=torch.bool)
    for component, index in COMPONENT_INDEX.items():
        rows = torch.nonzero(components == index, as_tuple=False).flatten()
        if rows.numel():
            columns = torch.tensor(LEGAL_VALUES[component], dtype=torch.long, device=logits.device)
            allowed[rows.unsqueeze(1), columns.unsqueeze(0)] = True
    if allowed_mask is not None:
        if allowed_mask.shape != allowed.shape:
            raise ValueError("stream-specific allowed_mask must match logits")
        allowed &= allowed_mask.to(device=logits.device, dtype=torch.bool)
    if not bool(allowed.any(dim=1).all()):
        raise ValueError("a typed query has no legal public output values")
    return logits.masked_fill(~allowed, torch.finfo(logits.dtype).min)


@dataclass(frozen=True, slots=True)
class MemoryExample:
    """One causal public prefix; `target` is used only by training code."""

    query_id: str
    stream_id: str
    slot: str
    features: torch.Tensor
    actions: torch.Tensor
    subject: int
    component: int
    allowed_mask: torch.Tensor
    target: int


class MemoryDataset(Dataset[MemoryExample]):
    def __init__(self, examples: list[MemoryExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> MemoryExample:
        return self.examples[index]


def collate_examples(rows: list[MemoryExample]) -> dict[str, torch.Tensor | list[str]]:
    if not rows:
        raise ValueError("cannot collate an empty batch")
    lengths = torch.tensor([row.features.shape[0] for row in rows], dtype=torch.long)
    max_length = int(lengths.max())
    features = torch.zeros(len(rows), max_length, rows[0].features.shape[1], dtype=torch.float32)
    actions = torch.zeros(len(rows), max_length, dtype=torch.long)
    allowed_mask = torch.stack([row.allowed_mask for row in rows])
    for index, row in enumerate(rows):
        features[index, : row.features.shape[0]] = row.features
        actions[index, : row.actions.shape[0]] = row.actions
    return {
        "query_ids": [row.query_id for row in rows],
        "slots": [row.slot for row in rows],
        "features": features,
        "actions": actions,
        "lengths": lengths,
        "subject": torch.tensor([row.subject for row in rows], dtype=torch.long),
        "component": torch.tensor([row.component for row in rows], dtype=torch.long),
        "allowed_mask": allowed_mask,
        "target": torch.tensor([row.target for row in rows], dtype=torch.long),
    }
