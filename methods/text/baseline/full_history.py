"""Full-History and fixed-window Full-History baselines."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from e1core.answer_codec import AnswerCodecError, UNTRACKED, answer_instruction, decode_answer, state_dump_instruction, unwrap_single_markdown_fence
from e1core.protocol import PublicEvent, QueryExample, REPRESENTATIONS
from e1core.transition_spec import public_transition_prompt

from .base import Generation, MemoryMethod, TextBackend, history_payload, state_digest


def _parse_state(raw: str) -> Mapping[str, Any]:
    value = json.loads(unwrap_single_markdown_fence(raw))
    if not isinstance(value, Mapping) or set(value) != {"state"} or not isinstance(value["state"], Mapping):
        raise AnswerCodecError("state dump must be exactly {'state': {...}}")
    return dict(value["state"])


class FullHistory(MemoryMethod):
    """Answer from a complete prefix, optionally retaining only its latest B events."""

    name = "full_history"

    def __init__(self, backend: TextBackend, *, representation: str = "structured", max_events: int | None = None) -> None:
        if max_events is not None and max_events <= 0:
            raise ValueError("max_events must be positive")
        if representation not in REPRESENTATIONS:
            raise ValueError(f"unsupported representation {representation!r}")
        self.backend, self.representation, self.max_events = backend, representation, max_events
        self.events: list[PublicEvent] = []
        self.system = "\n\n".join(
            ["You are a precise world-state reasoning model. Use only the public event history. Do not invent facts.", public_transition_prompt(representation)]
        )

    def reset(self) -> None:
        self.events = []

    def _view(self) -> list[PublicEvent]:
        return self.events if self.max_events is None else self.events[-self.max_events :]

    def step(self, event: PublicEvent) -> Sequence[Generation]:
        self.events.append(event)
        return ()

    def answer(self, example: QueryExample) -> tuple[Any, Sequence[Generation]]:
        user = "\n".join(
            [
                "Public event history:",
                json.dumps(history_payload(self._view()), ensure_ascii=False, sort_keys=True),
                "Query:",
                json.dumps(
                    {"entity_id": example.target_entity_id, "component": example.target_component},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                answer_instruction(example.target_component),
            ]
        )
        call = self.backend.generate(role="history_answer", system=self.system, user=user, max_new_tokens=96)
        try:
            return decode_answer(call.raw, component=example.target_component), (call,)
        except AnswerCodecError:
            return UNTRACKED, (call,)

    def dump(self, scope: Sequence[str]) -> tuple[Mapping[str, Any], Sequence[Generation]]:
        user = "\n".join(
            [
                "Public event history:",
                json.dumps(history_payload(self._view()), ensure_ascii=False, sort_keys=True),
                state_dump_instruction(scope),
            ]
        )
        call = self.backend.generate(role="history_state_dump", system=self.system, user=user, max_new_tokens=256)
        try:
            return _parse_state(call.raw), (call,)
        except (AnswerCodecError, json.JSONDecodeError):
            return {}, (call,)

    def trace(self) -> Mapping[str, Any]:
        return {
            "method": self.name,
            "representation": self.representation,
            "retained_event_count": len(self._view()),
            "observed_event_count": len(self.events),
            "history_digest": state_digest({str(i): event.to_dict() for i, event in enumerate(self._view())}),
        }


class BudgetedFullHistory(FullHistory):
    name = "budgeted_full_history"

    def __init__(self, backend: TextBackend, *, max_events: int, representation: str = "structured") -> None:
        super().__init__(backend, representation=representation, max_events=max_events)
