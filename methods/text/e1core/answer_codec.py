"""One typed answer/state codec shared by every StateReturn Text method.

The codec intentionally does not repair semantic mistakes through synonyms or
world knowledge.  It only canonicalizes a valid JSON object into the exact
public E1 value type; malformed model output is measured as invalid.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping

from .protocol import BASE_COMPONENTS, E1ProtocolError


UNTRACKED = "__UNTRACKED__"
CODEC_VERSION = "statereturn-text-answer-codec/v1"


class AnswerCodecError(E1ProtocolError):
    """A method answer/state dump does not match the component's public type."""


def unwrap_single_markdown_fence(raw: str) -> str:
    """Remove at most one outer code fence, without repairing any content.

    This is a representation- and method-neutral transport normalization: a
    local instruction model often wraps otherwise-valid JSON/DSL in a fence.
    It does not accept surrounding prose, multiple blocks, malformed JSON, or
    semantic aliases.
    """

    if not isinstance(raw, str):
        raise AnswerCodecError("model output must be text")
    source = raw.strip()
    if not source.startswith("```"):
        return source
    lines = source.splitlines()
    if len(lines) < 3 or not lines[0].startswith("```") or lines[-1].strip() != "```":
        raise AnswerCodecError("malformed markdown code fence")
    if any(line.strip().startswith("```") for line in lines[1:-1]):
        raise AnswerCodecError("only one outer markdown code fence is allowed")
    return "\n".join(lines[1:-1]).strip()


def _entity_id(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise AnswerCodecError(f"{field} must be a non-empty trimmed entity ID")
    return value


def normalize_value(component: str, value: Any) -> Any:
    """Validate one prediction without seeing a target label."""

    if component == "openness":
        if not isinstance(value, str) or value not in {"open", "closed"}:
            raise AnswerCodecError("openness answer must be 'open' or 'closed'")
        return value
    if component == "toggle_state":
        if not isinstance(value, str) or value not in {"on", "off"}:
            raise AnswerCodecError("toggle_state answer must be 'on' or 'off'")
        return value
    if component == "place":
        if not isinstance(value, Mapping) or set(value) != {"kind", "target"}:
            raise AnswerCodecError("place answer must be exactly {'kind', 'target'}")
        kind, target = value.get("kind"), value.get("target")
        if kind not in {"on", "inside", "held_by"}:
            raise AnswerCodecError("place.kind must be on, inside, or held_by")
        return {"kind": str(kind), "target": _entity_id(target, field="place.target")}
    raise AnswerCodecError(f"unsupported component {component!r}; expected one of {BASE_COMPONENTS}")


def decode_answer(raw: str, *, component: str) -> Any:
    """Decode a strict model response: exactly ``{"answer": <typed value>}``."""

    try:
        value = json.loads(unwrap_single_markdown_fence(raw))
    except json.JSONDecodeError as exc:
        raise AnswerCodecError("answer must be valid JSON after optional outer code-fence removal") from exc
    if not isinstance(value, Mapping) or set(value) != {"answer"}:
        raise AnswerCodecError("answer must be exactly {'answer': <value>}")
    return normalize_value(component, value["answer"])


def encode_answer(component: str, value: Any) -> str:
    """Return the unique JSON representation required in LLM answer prompts."""

    return json.dumps({"answer": normalize_value(component, value)}, ensure_ascii=False, sort_keys=True)


def answer_instruction(component: str) -> str:
    """Component-specific output contract injected verbatim into all LLM methods."""

    if component == "openness":
        return 'Return exactly one JSON object: {"answer":"open"} or {"answer":"closed"}. A single outer Markdown code fence is allowed, but no prose.'
    if component == "toggle_state":
        return 'Return exactly one JSON object: {"answer":"on"} or {"answer":"off"}. A single outer Markdown code fence is allowed, but no prose.'
    if component == "place":
        return (
            'Return exactly one valid JSON object. The value of answer must be an object with exactly '
            'kind and target. The value of kind must be exactly one of "on", "inside", or "held_by"; '
            'target must be a stable entity ID. Example: {"answer":{"kind":"on","target":"table_1"}}. '
            'A single outer Markdown code fence is allowed, but no prose.'
        )
    raise AnswerCodecError(f"unsupported component {component!r}")


def state_dump_instruction(scope: Iterable[str]) -> str:
    """Return the frozen typed-output contract for every full-state probe."""

    keys = list(scope)
    return "\n".join(
        [
            "Return exactly one valid JSON object with the form {\"state\": {\"entity|component\": value, ...}}.",
            "Return only facts your memory currently tracks. You may use only these public candidate keys:",
            json.dumps(keys, ensure_ascii=False),
            "Value types:",
            '- openness must be "open" or "closed".',
            '- toggle_state must be "on" or "off".',
            '- place must be exactly one of {"kind":"on","target":"entity_id"}, '
            '{"kind":"inside","target":"entity_id"}, or {"kind":"held_by","target":"agent_1"}.',
            "Example: {\"state\": {\"drawer_1|openness\": \"open\", \"mug_1|place\": {\"kind\": \"on\", \"target\": \"table_1\"}}}.",
            "A single outer Markdown code fence is allowed, but no prose.",
        ]
    )


def parse_slot_key(key: Any) -> tuple[str, str]:
    if not isinstance(key, str) or "|" not in key:
        raise AnswerCodecError("state key must be '<entity_id>|<component>'")
    entity_id, component = key.rsplit("|", 1)
    return _entity_id(entity_id, field="state entity_id"), component


def normalize_state_dump(value: Any, *, allowed_slots: Iterable[str] | None = None) -> dict[str, Any]:
    """Validate a predicted tracked-state dump, retaining no invalid values.

    ``allowed_slots`` is public scope metadata produced from the input history.
    An extra slot is rejected rather than silently ignored.
    """

    if not isinstance(value, Mapping):
        raise AnswerCodecError("tracked state dump must be an object")
    allowed = None if allowed_slots is None else set(allowed_slots)
    result: dict[str, Any] = {}
    for key, item in value.items():
        entity_id, component = parse_slot_key(key)
        canonical_key = f"{entity_id}|{component}"
        if allowed is not None and canonical_key not in allowed:
            raise AnswerCodecError(f"state dump contains an out-of-scope slot: {canonical_key}")
        if item == UNTRACKED:
            result[canonical_key] = UNTRACKED
        else:
            result[canonical_key] = normalize_value(component, item)
    return dict(sorted(result.items()))
