"""Shared method interface, public scope extraction, and generation records."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Protocol, Sequence

from e1core.protocol import BASE_COMPONENTS, NaturalEvent, PublicEvent, QueryExample, StructuredEvent, sha256_json


UNTRACKED = "__UNTRACKED__"


@dataclass(frozen=True, slots=True)
class Generation:
    role: str
    raw: str
    seconds: float
    input_tokens: int | None = None
    output_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "raw": self.raw,
            "generation_seconds": round(self.seconds, 6),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


class TextBackend(Protocol):
    name: str

    def generate(self, *, role: str, system: str, user: str, max_new_tokens: int) -> Generation: ...


class MemoryMethod(abc.ABC):
    """All methods process only one current public event at a time."""

    name: str

    @abc.abstractmethod
    def reset(self) -> None: ...

    @abc.abstractmethod
    def step(self, event: PublicEvent) -> Sequence[Generation]: ...

    @abc.abstractmethod
    def answer(self, example: QueryExample) -> tuple[Any, Sequence[Generation]]: ...

    @abc.abstractmethod
    def dump(self, scope: Sequence[str]) -> tuple[Mapping[str, Any], Sequence[Generation]]: ...

    @abc.abstractmethod
    def trace(self) -> Mapping[str, Any]: ...

    def compute_usage(self) -> Mapping[str, Any]:
        """Optional cumulative non-generative compute accounting for a method."""

        return {}


def public_scope(history: Iterable[PublicEvent], *, query_entity_id: str | None = None) -> tuple[str, ...]:
    """Return a public *candidate* state-dump scope, not an Oracle slot list.

    Structured actions publicly expose their changed component.  Natural text
    does not: inferring it with the template regex would turn the probe scope
    into a hidden deterministic parser.  Therefore Natural runs receive the
    representation-neutral Cartesian candidate set of visible entities and
    Base components; metrics later project it to evaluator-only tracked slots.
    """

    action_component = {
        "OpenObject": "openness",
        "CloseObject": "openness",
        "ToggleObjectOn": "toggle_state",
        "ToggleObjectOff": "toggle_state",
        "PickupObject": "place",
        "PutObject": "place",
    }
    slots: set[str] = set()
    natural_entities: set[str] = set()
    for event in history:
        if isinstance(event, StructuredEvent):
            component = action_component.get(event.action_type)
            if component is not None and event.outcome == "success_changed":
                slots.add(f"{event.arguments['object_id']}|{component}")
        elif isinstance(event, NaturalEvent):
            natural_entities.update(entity for entity in event.entity_ids() if entity != "agent_1")
    if natural_entities:
        if query_entity_id and query_entity_id != "agent_1":
            natural_entities.add(query_entity_id)
        slots.update(f"{entity}|{component}" for entity in natural_entities for component in BASE_COMPONENTS)
    return tuple(sorted(slots))


def history_payload(events: Iterable[PublicEvent]) -> list[dict[str, Any]]:
    return [event.to_dict() for event in events]


def call_usage(calls: Iterable[Generation]) -> dict[str, Any]:
    calls = list(calls)
    known_input = [call.input_tokens for call in calls if isinstance(call.input_tokens, int)]
    known_output = [call.output_tokens for call in calls if isinstance(call.output_tokens, int)]
    return {
        "agent_call_count": len(calls),
        "input_tokens": sum(known_input) if known_input else None,
        "output_tokens": sum(known_output) if known_output else None,
        "model_generation_seconds": round(sum(call.seconds for call in calls), 6),
    }


def state_digest(state: Mapping[str, Any]) -> str:
    return sha256_json(dict(sorted(state.items())))
