"""Bounded natural-language rolling-summary baseline."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from e1core.answer_codec import AnswerCodecError, UNTRACKED, answer_instruction, decode_answer, state_dump_instruction
from e1core.protocol import PublicEvent, QueryExample, REPRESENTATIONS
from e1core.transition_spec import public_transition_prompt

from .base import Generation, MemoryMethod, TextBackend, state_digest
from .full_history import _parse_state


class RollingSummary(MemoryMethod):
    name = "rolling_summary"

    def __init__(self, backend: TextBackend, *, representation: str = "structured", max_chars: int = 1600) -> None:
        if max_chars < 128:
            raise ValueError("max_chars must be at least 128")
        if representation not in REPRESENTATIONS:
            raise ValueError(f"unsupported representation {representation!r}")
        self.backend, self.representation, self.max_chars = backend, representation, max_chars
        self.system = "\n\n".join(
            ["You maintain a bounded natural-language world memo from public events only.", public_transition_prompt(representation)]
        )
        self.summary = "No public state has been observed yet."
        self.event_count = 0

    def reset(self) -> None:
        self.summary = "No public state has been observed yet."
        self.event_count = 0

    def step(self, event: PublicEvent) -> Sequence[Generation]:
        user = "\n".join(
            [
                f"Current memo (maximum {self.max_chars} characters):",
                self.summary,
                "New public event:",
                json.dumps(event.to_dict(), ensure_ascii=False, sort_keys=True),
                "Write a replacement natural-language memo. Keep only current supported facts; overwrite obsolete values; do not claim a state change for failed/no-op events; do not use Oracle labels or hidden state.",
            ]
        )
        call = self.backend.generate(role="summary_update", system=self.system, user=user, max_new_tokens=384)
        self.summary = call.raw.strip()[: self.max_chars] or self.summary
        self.event_count += 1
        return (call,)

    def answer(self, example: QueryExample) -> tuple[Any, Sequence[Generation]]:
        user = "\n".join(
            [
                "Current rolling memo:", self.summary,
                "Query:",
                json.dumps({"entity_id": example.target_entity_id, "component": example.target_component}, sort_keys=True),
                answer_instruction(example.target_component),
            ]
        )
        call = self.backend.generate(role="summary_answer", system=self.system, user=user, max_new_tokens=96)
        try:
            return decode_answer(call.raw, component=example.target_component), (call,)
        except AnswerCodecError:
            return UNTRACKED, (call,)

    def dump(self, scope: Sequence[str]) -> tuple[Mapping[str, Any], Sequence[Generation]]:
        user = "\n".join(
            [
                "Current rolling memo:", self.summary,
                state_dump_instruction(scope),
            ]
        )
        call = self.backend.generate(role="summary_state_dump", system=self.system, user=user, max_new_tokens=256)
        try:
            return _parse_state(call.raw), (call,)
        except (AnswerCodecError, json.JSONDecodeError):
            return {}, (call,)

    def trace(self) -> Mapping[str, Any]:
        return {
            "method": self.name,
            "representation": self.representation,
            "observed_event_count": self.event_count,
            "summary_char_count": len(self.summary),
            "summary_sha256": state_digest({"summary": self.summary}),
        }
