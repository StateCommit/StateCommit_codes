"""Public dual-view Base-Track input contract for StateReturn v1.0.

Every E1 method opens exactly one public representation at a time.  In
particular, a Natural run never opens the aligned Structured record to recover
an action name, a placement effect, or an answer.  The typed query is shared
by design; only the event-history representation changes.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, TypeAlias


DATASET_VERSION = "StateReturn/v1.0"
PROTOCOL_VERSION = "statereturn-e1-dual-view-base-track/v1"
BASE_COMPONENTS = ("openness", "place", "toggle_state")
REPRESENTATIONS = ("structured", "natural")
_BASE_COMPONENT_SET = frozenset(BASE_COMPONENTS)
SPLITS = ("train", "validation", "test")
OUTCOMES = frozenset({"success_changed", "success_no_change", "failure_no_change"})
_NATURAL_OUTCOMES = {
    "The interaction completed successfully.": "success_changed",
    "The interaction completed, but no observable world change followed.": "success_no_change",
    "The interaction failed. No change to the world was observed.": "failure_no_change",
}
_ENTITY_RE = r"(?P<entity>[A-Za-z][A-Za-z0-9]*_[0-9]+)"


class E1ProtocolError(ValueError):
    """Raised when a public v2.4 record violates the frozen E1 contract."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise E1ProtocolError(f"cannot read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise E1ProtocolError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise E1ProtocolError(f"missing JSONL: {source}")
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line:
            raise E1ProtocolError(f"blank line in {source}:{line_no}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise E1ProtocolError(f"invalid JSON in {source}:{line_no}") from exc
        if not isinstance(row, dict):
            raise E1ProtocolError(f"JSONL row must be object in {source}:{line_no}")
        rows.append(row)
    return rows


def _required_id(value: Any, *, field: str, context: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise E1ProtocolError(f"{context}: {field} must be a non-empty trimmed string")
    return value


@dataclass(frozen=True, slots=True)
class PlaceEffect:
    """Minimal public Structured execution observation for one place update."""

    entity_id: str
    kind: str
    target: str

    @classmethod
    def from_public(cls, value: Any, *, action_type: str, arguments: Mapping[str, Any], context: str) -> "PlaceEffect":
        if not isinstance(value, Mapping) or set(value) != {"entity_id", "kind", "target"}:
            raise E1ProtocolError(f"{context}: place_effect must have exactly entity_id/kind/target")
        effect = cls(
            entity_id=_required_id(value.get("entity_id"), field="place_effect.entity_id", context=context),
            kind=_required_id(value.get("kind"), field="place_effect.kind", context=context),
            target=_required_id(value.get("target"), field="place_effect.target", context=context),
        )
        if effect.entity_id != arguments.get("object_id"):
            raise E1ProtocolError(f"{context}: place_effect entity differs from arguments.object_id")
        if action_type == "PutObject":
            if effect.kind not in {"on", "inside"} or effect.target != arguments.get("destination_id"):
                raise E1ProtocolError(f"{context}: invalid PutObject place_effect")
        elif action_type == "PickupObject":
            if effect.kind != "held_by" or effect.target != "agent_1":
                raise E1ProtocolError(f"{context}: invalid PickupObject place_effect")
        else:
            raise E1ProtocolError(f"{context}: place_effect is only legal for PutObject/PickupObject")
        return effect

    def to_dict(self) -> dict[str, str]:
        return {"entity_id": self.entity_id, "kind": self.kind, "target": self.target}


@dataclass(frozen=True, slots=True)
class StructuredEvent:
    event_id: str
    action_type: str
    arguments: Mapping[str, Any]
    outcome: str
    place_effect: PlaceEffect | None

    @classmethod
    def from_public(cls, value: Any, *, context: str) -> "StructuredEvent":
        if not isinstance(value, Mapping):
            raise E1ProtocolError(f"{context}: history event must be an object")
        event_id = _required_id(value.get("event_id"), field="event_id", context=context)
        action_type = _required_id(value.get("action_type"), field="action_type", context=context)
        arguments, outcome = value.get("arguments"), value.get("outcome")
        if not isinstance(arguments, Mapping):
            raise E1ProtocolError(f"{context}: arguments must be an object")
        arguments = dict(arguments)
        if outcome not in OUTCOMES:
            raise E1ProtocolError(f"{context}: unsupported outcome {outcome!r}")
        _required_id(arguments.get("object_id"), field="arguments.object_id", context=context)
        if action_type == "PutObject":
            _required_id(arguments.get("destination_id"), field="arguments.destination_id", context=context)
        raw_effect = value.get("place_effect")
        needs_effect = action_type in {"PutObject", "PickupObject"} and outcome == "success_changed"
        if needs_effect != isinstance(raw_effect, Mapping):
            raise E1ProtocolError(f"{context}: place_effect presence disagrees with action/outcome")
        effect = None if raw_effect is None else PlaceEffect.from_public(raw_effect, action_type=action_type, arguments=arguments, context=context)
        allowed = {"event_id", "action_type", "arguments", "outcome"}
        if effect is not None:
            allowed.add("place_effect")
        if set(value) != allowed:
            raise E1ProtocolError(f"{context}: unexpected public event fields {sorted(set(value) - allowed)}")
        return cls(event_id, action_type, arguments, str(outcome), effect)

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {"event_id": self.event_id, "action_type": self.action_type, "arguments": dict(self.arguments), "outcome": self.outcome}
        if self.place_effect is not None:
            value["place_effect"] = self.place_effect.to_dict()
        return value

    def entity_ids(self) -> tuple[str, ...]:
        values = {"agent_1", str(self.arguments["object_id"])}
        if self.action_type == "PutObject":
            values.add(str(self.arguments["destination_id"]))
        if self.place_effect is not None:
            values.add(self.place_effect.target)
        return tuple(sorted(values))


@dataclass(frozen=True, slots=True)
class NaturalEvent:
    """The public Natural input: no action API, arguments, or place effect."""

    event_id: str
    description: str
    outcome_text: str

    @classmethod
    def from_public(cls, value: Any, *, context: str) -> "NaturalEvent":
        if not isinstance(value, Mapping) or set(value) != {"event_id", "description", "outcome_text"}:
            raise E1ProtocolError(f"{context}: Natural event must expose exactly event_id/description/outcome_text")
        event_id = _required_id(value.get("event_id"), field="event_id", context=context)
        description = _required_id(value.get("description"), field="description", context=context)
        outcome_text = _required_id(value.get("outcome_text"), field="outcome_text", context=context)
        if outcome_text not in _NATURAL_OUTCOMES:
            raise E1ProtocolError(f"{context}: unsupported Natural outcome sentence")


        forbidden = ("OpenObject", "CloseObject", "ToggleObject", "PutObject", "PickupObject", "place_effect")
        if any(token in description or token in outcome_text for token in forbidden):
            raise E1ProtocolError(f"{context}: Natural event leaks canonical-action vocabulary")
        return cls(event_id, description, outcome_text)

    @property
    def outcome(self) -> str:
        return _NATURAL_OUTCOMES[self.outcome_text]

    def to_dict(self) -> dict[str, str]:
        return {"event_id": self.event_id, "description": self.description, "outcome_text": self.outcome_text}

    def entity_ids(self) -> tuple[str, ...]:
        return tuple(sorted({"agent_1", *re.findall(r"[A-Za-z][A-Za-z0-9]*_[0-9]+", self.description)}))


PublicEvent: TypeAlias = StructuredEvent | NaturalEvent


@dataclass(frozen=True, slots=True)
class QueryExample:
    query_id: str
    episode_id: str
    split: str
    checkpoint_k: int
    history: tuple[PublicEvent, ...]
    target_entity_id: str
    target_component: str
    representation: str = "structured"

    def public_fingerprint(self) -> str:
        return sha256_json({
            "representation": self.representation,
            "query_id": self.query_id,
            "episode_id": self.episode_id,
            "split": self.split,
            "checkpoint_k": self.checkpoint_k,
            "history": [event.to_dict() for event in self.history],
            "query": {"entity_id": self.target_entity_id, "component": self.target_component},
        })


def _parse_example(row: Mapping[str, Any], *, source: Path, representation: str) -> QueryExample:
    context = f"{source}:{row.get('record_id', '<unknown>')}"
    episode_id = _required_id(row.get("episode_id"), field="episode_id", context=context)
    split, checkpoint_k = row.get("split"), row.get("checkpoint_k")
    query, history = row.get("query"), row.get("history")
    if split not in SPLITS or not isinstance(checkpoint_k, int) or checkpoint_k < 0:
        raise E1ProtocolError(f"{context}: invalid split/checkpoint")
    if not isinstance(query, Mapping) or not isinstance(history, list):
        raise E1ProtocolError(f"{context}: query/history are malformed")
    query_id = _required_id(query.get("query_id"), field="query.query_id", context=context)
    target_entity_id = _required_id(query.get("entity_id"), field="query.entity_id", context=context)
    target_component = query.get("component")
    if not isinstance(target_component, str) or not target_component:
        raise E1ProtocolError(f"{context}: invalid query component")
    if set(query) != {"query_id", "entity_id", "component"}:
        raise E1ProtocolError(f"{context}: unexpected query fields")
    if row.get("record_id") != f"{representation}:{query_id}":
        raise E1ProtocolError(f"{context}: record_id/query_id mismatch")
    parser = StructuredEvent.from_public if representation == "structured" else NaturalEvent.from_public
    events = tuple(parser(item, context=context) for item in history)
    if not events or len({event.event_id for event in events}) != len(events):
        raise E1ProtocolError(f"{context}: empty or duplicate event history")
    return QueryExample(query_id, episode_id, str(split), checkpoint_k, events, target_entity_id, target_component, representation)


def load_examples(
    dataset_root: str | Path, *, split: str, representation: str = "structured", components: Iterable[str] = BASE_COMPONENTS,
) -> list[QueryExample]:
    """Load one public StateReturn v1.0 Base-Track representation."""

    if split not in SPLITS or representation not in REPRESENTATIONS:
        raise E1ProtocolError(f"unsupported split/representation: {split!r}/{representation!r}")
    allowed = tuple(str(item) for item in components)
    if not allowed or not set(allowed).issubset(_BASE_COMPONENT_SET):
        raise E1ProtocolError(f"components must be a nonempty subset of {BASE_COMPONENTS}")
    source = Path(dataset_root) / "data" / representation / f"{split}.jsonl"
    examples = [_parse_example(row, source=source, representation=representation) for row in read_jsonl(source)]
    selected = [example for example in examples if example.target_component in allowed]
    if len({example.query_id for example in selected}) != len(selected):
        raise E1ProtocolError(f"duplicate query IDs in {source}")
    validate_prefixes(selected)
    return selected


def validate_prefixes(examples: Iterable[QueryExample]) -> None:
    by_episode: dict[str, list[QueryExample]] = defaultdict(list)
    for example in examples:
        by_episode[example.episode_id].append(example)
    for episode_id, rows in by_episode.items():
        previous: QueryExample | None = None
        source_length: int | None = None
        representation: str | None = None
        for row in sorted(rows, key=lambda item: item.checkpoint_k):
            if representation is None:
                representation = row.representation
            elif representation != row.representation:
                raise E1ProtocolError(f"episode {episode_id}: mixed representations")
            if row.checkpoint_k == 0:
                if source_length is not None:
                    raise E1ProtocolError(f"episode {episode_id}: duplicate K=0")
                source_length = len(row.history)
            elif source_length is None or len(row.history) != source_length + row.checkpoint_k:
                raise E1ProtocolError(f"episode {episode_id}: invalid K-prefix length at K={row.checkpoint_k}")
            if previous is not None:
                old = tuple(event.event_id for event in previous.history)
                new = tuple(event.event_id for event in row.history)
                if new[: len(old)] != old:
                    raise E1ProtocolError(f"episode {episode_id}: K prefixes are not append-only")
            previous = row


def _verify_checksums(root: Path) -> int:
    source = root / "metadata" / "checksums.sha256"
    if not source.is_file():
        raise E1ProtocolError("public release lacks checksums")
    count = 0
    for line in source.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        target = root / relative
        if not target.is_file() or sha256_file(target) != digest:
            raise E1ProtocolError(f"checksum mismatch: {relative}")
        count += 1
    return count


def verify_release(dataset_root: str | Path) -> dict[str, Any]:
    """Verify only the StateReturn v1.0 public boundary."""

    root = Path(dataset_root)
    manifest = read_json(root / "metadata" / "release_manifest.json")
    if (
        manifest.get("benchmark") != "StateReturn-Text"
        or manifest.get("release_version") != DATASET_VERSION
        or manifest.get("package_kind") != "public"
        or manifest.get("representations") != ["structured", "natural"]
    ):
        raise E1ProtocolError("E1 requires the StateReturn v1.0 public Text release")
    if (root / "labels" / "test.jsonl").exists():
        raise E1ProtocolError("public release must never contain test labels")
    contract = manifest.get("natural_transition_contract")
    if not isinstance(contract, Mapping):
        raise E1ProtocolError("public release lacks the Natural transition contract")
    contract_path = contract.get("path")
    if contract_path != "schema/natural_event_transition_spec_v1.json":
        raise E1ProtocolError("Natural transition contract has an unexpected path")
    local_contract = root / str(contract_path)
    if not local_contract.is_file() or sha256_file(local_contract) != contract.get("sha256"):
        raise E1ProtocolError("Natural transition contract checksum mismatch")
    if read_json(local_contract).get("spec_version") != contract.get("spec_version"):
        raise E1ProtocolError("Natural transition contract version mismatch")
    report: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset_version": DATASET_VERSION,
        "public_checksum_count": _verify_checksums(root),
        "representations": {},
        "status": "passed",
    }
    public_ids: dict[str, set[str]] = {}
    for representation in REPRESENTATIONS:
        report["representations"][representation] = {}
        for split in SPLITS:
            examples = load_examples(root, split=split, representation=representation)
            ids = {example.query_id for example in examples}
            if representation == "structured":
                public_ids[split] = ids
            elif ids != public_ids[split]:
                raise E1ProtocolError(f"dual-view query-ID mismatch for {split}")
            report["representations"][representation][split] = {
                "base_query_count": len(examples),
                "episode_count": len({example.episode_id for example in examples}),
                "k_values": sorted({example.checkpoint_k for example in examples}),
                "components": {component: sum(example.target_component == component for example in examples) for component in BASE_COMPONENTS},
            }
    return report
