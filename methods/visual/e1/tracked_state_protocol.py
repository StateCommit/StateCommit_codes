"""Observed/Tracked Active-State protocol for Visual WCM E1.

The frozen G6 labels contain the *complete simulator state*.  That is not an
appropriate full-state target: a causal method should only be asked to report
state slots whose changes occurred in its public RGB/action prefix.  This
module defines the method-visible slot scope for each state query.  Values
remain evaluator-private.

The protocol is a sidecar to G6 rather than a mutation of the frozen dataset.
Methods may load only ``<sidecar>/public/tracked_state_scopes.json``.  The
private companion is deliberately loaded by :mod:`e1.tracked_state_evaluator`
only after a complete prediction file exists.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from e0.predictions import validate_state_answer
from e0.public_protocol import PublicBenchmark, PublicProtocolError, PublicQuery


TRACKED_STATE_PROTOCOL_VERSION = "wcm-grid-state-evolution/v2-e1-tracked-state-v1"
TRACKED_STATE_TASK = "tracked_state_query"


class TrackedStateProtocolError(ValueError):
    """The tracked-state sidecar or a method prediction is malformed."""


def slot_key(entity_id: str, component: str) -> str:
    if not isinstance(entity_id, str) or not entity_id or "|" in entity_id:
        raise TrackedStateProtocolError("entity ID in a slot must be a non-empty string without '|'")
    if component not in {"place", "openness", "toggle_state"}:
        raise TrackedStateProtocolError(f"unsupported tracked-state component: {component!r}")
    return f"{entity_id}|{component}"


def parse_slot_key(value: object) -> tuple[str, str]:
    if not isinstance(value, str) or value.count("|") != 1:
        raise TrackedStateProtocolError("tracked slot must use exactly 'entity_id|component'")
    entity_id, component = value.split("|", 1)
    canonical = slot_key(entity_id, component)
    if canonical != value:
        raise TrackedStateProtocolError("tracked slot is not canonical")
    return entity_id, component


def _read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise TrackedStateProtocolError(f"cannot read tracked-state JSON {path}: {error}") from error


@dataclass(frozen=True, slots=True)
class TrackedStateScope:
    """One public declaration of the slots a state dump must contain."""

    query_id: str
    stream_id: str
    split: str
    k: int
    prefix_end_frame: int
    target_slot: str
    tracked_slots: tuple[str, ...]

    @classmethod
    def from_dict(cls, data: object, *, benchmark: PublicBenchmark) -> "TrackedStateScope":
        if not isinstance(data, dict):
            raise TrackedStateProtocolError("tracked-state scope must be an object")
        if set(data) != {"query_id", "stream_id", "split", "k", "prefix_end_frame", "target_slot", "tracked_slots"}:
            raise TrackedStateProtocolError("tracked-state scope has unsupported or missing fields")
        query_id = data["query_id"]
        if not isinstance(query_id, str) or query_id not in benchmark.queries:
            raise TrackedStateProtocolError("tracked-state scope refers to an unknown public query")
        query = benchmark.queries[query_id]
        if query.task != "state_query":
            raise TrackedStateProtocolError("tracked-state scopes apply only to state_query records")
        expected = {
            "stream_id": query.stream_id,
            "split": query.split,
            "k": query.k,
            "prefix_end_frame": query.prefix_end_frame,
            "target_slot": slot_key(query.entity_id, query.component),
        }
        for field, expected_value in expected.items():
            if data[field] != expected_value:
                raise TrackedStateProtocolError(f"tracked-state scope {field} does not match its public query")
        rows = data["tracked_slots"]
        if not isinstance(rows, list) or not rows or any(not isinstance(row, str) for row in rows):
            raise TrackedStateProtocolError("tracked_slots must be a non-empty string list")
        slots = tuple(rows)
        if len(set(slots)) != len(slots):
            raise TrackedStateProtocolError("tracked_slots must be unique")
        for row in slots:
            entity_id, component = parse_slot_key(row)
            catalog = {str(entry["entity_id"]): str(entry["entity_type"]) for entry in benchmark.streams[query.stream_id].entity_catalog}
            entity_type = catalog.get(entity_id)
            if entity_type is None:
                raise TrackedStateProtocolError("tracked slot entity is absent from the stream public catalog")
            if (component == "place" and entity_type not in {"ball", "key"}) or (
                component == "openness" and entity_type != "door"
            ) or (component == "toggle_state" and entity_type != "switch"):
                raise TrackedStateProtocolError("tracked slot component is incompatible with its public entity type")
        target_slot = expected["target_slot"]
        if target_slot not in slots:
            raise TrackedStateProtocolError("the target state slot must be included in tracked_slots")
        return cls(
            query_id=query_id,
            stream_id=query.stream_id,
            split=query.split,
            k=query.k,
            prefix_end_frame=query.prefix_end_frame,
            target_slot=target_slot,
            tracked_slots=slots,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "stream_id": self.stream_id,
            "split": self.split,
            "k": self.k,
            "prefix_end_frame": self.prefix_end_frame,
            "target_slot": self.target_slot,
            "tracked_slots": list(self.tracked_slots),
        }


@dataclass(frozen=True, slots=True)
class PublicTrackedStateScopes:
    """Validated public sidecar indexed by state-query ID."""

    public_root: Path
    scopes: dict[str, TrackedStateScope]

    def for_query(self, query: PublicQuery) -> TrackedStateScope:
        try:
            scope = self.scopes[query.query_id]
        except KeyError as error:
            raise TrackedStateProtocolError(f"missing tracked-state scope: {query.query_id}") from error
        if scope.stream_id != query.stream_id:
            raise TrackedStateProtocolError("tracked-state scope/query stream mismatch")
        return scope

    def for_split(self, split: str) -> tuple[TrackedStateScope, ...]:
        return tuple(scope for scope in self.scopes.values() if scope.split == split)


def load_public_tracked_state_scopes(sidecar_public_root: str | Path, benchmark: PublicBenchmark) -> PublicTrackedStateScopes:
    """Load only the method-visible tracked slot declarations."""

    root = Path(sidecar_public_root).resolve()
    if root.name != "public":
        raise TrackedStateProtocolError("tracked-state loader requires the sidecar public directory exactly")
    document = _read_json(root / "tracked_state_scopes.json")
    if not isinstance(document, dict) or set(document) != {"schema_version", "source_benchmark_schema_version", "scope_count", "scopes"}:
        raise TrackedStateProtocolError("tracked-state public sidecar has invalid top-level fields")
    if document["schema_version"] != TRACKED_STATE_PROTOCOL_VERSION:
        raise TrackedStateProtocolError("unexpected tracked-state sidecar version")


    if document["source_benchmark_schema_version"] != "wcm-grid-state-evolution/v2-formal-v1":
        raise TrackedStateProtocolError("tracked-state sidecar was built for another benchmark version")
    rows = document["scopes"]
    if not isinstance(rows, list) or document["scope_count"] != len(rows):
        raise TrackedStateProtocolError("tracked-state public scope_count mismatch")
    scopes: dict[str, TrackedStateScope] = {}
    for row in rows:
        scope = TrackedStateScope.from_dict(row, benchmark=benchmark)
        if scope.query_id in scopes:
            raise TrackedStateProtocolError("duplicate tracked-state scope query ID")
        scopes[scope.query_id] = scope
    expected = {query.query_id for query in benchmark.queries.values() if query.task == "state_query"}
    if set(scopes) != expected:
        raise TrackedStateProtocolError("tracked-state scopes must exactly cover public state queries")
    return PublicTrackedStateScopes(public_root=root, scopes=scopes)


@dataclass(frozen=True, slots=True)
class TrackedStatePrediction:
    query_id: str
    state: dict[str, str | dict[str, str]]
    untracked_slots: frozenset[str] = frozenset()


def parse_tracked_state_prediction(record: object, *, scope: TrackedStateScope) -> TrackedStatePrediction:
    """Strictly parse a public method's complete active-state dump."""

    if not isinstance(record, dict) or set(record) not in (
        {"query_id", "task", "state"},
        {"query_id", "task", "state", "untracked_slots"},
    ):
        raise TrackedStateProtocolError("tracked-state prediction requires query_id, task, state, and optional untracked_slots")
    if record.get("query_id") != scope.query_id or record.get("task") != TRACKED_STATE_TASK:
        raise TrackedStateProtocolError("tracked-state prediction identity/task mismatch")
    state = record.get("state")
    if not isinstance(state, dict) or set(state) != set(scope.tracked_slots):
        raise TrackedStateProtocolError("tracked-state prediction must contain exactly the declared tracked slots")
    missing = record.get("untracked_slots", [])
    if not isinstance(missing, list) or any(not isinstance(slot, str) for slot in missing):
        raise TrackedStateProtocolError("untracked_slots must be a string list when present")
    if len(set(missing)) != len(missing) or not set(missing) <= set(scope.tracked_slots):
        raise TrackedStateProtocolError("untracked_slots must be a unique subset of tracked_slots")
    normalized: dict[str, str | dict[str, str]] = {}
    for slot in scope.tracked_slots:
        _, component = parse_slot_key(slot)
        try:
            normalized[slot] = validate_state_answer(component, state[slot])
        except PublicProtocolError as error:
            raise TrackedStateProtocolError(f"invalid typed value for tracked slot {slot}") from error
    return TrackedStatePrediction(query_id=scope.query_id, state=normalized, untracked_slots=frozenset(missing))


def read_tracked_state_predictions(path: str | Path, scopes: Iterable[TrackedStateScope]) -> dict[str, TrackedStatePrediction]:
    """Read exactly one strict state dump for every selected public query."""

    indexed = {scope.query_id: scope for scope in scopes}
    if not indexed:
        raise TrackedStateProtocolError("cannot read predictions for an empty tracked-state scope")
    predictions: dict[str, TrackedStatePrediction] = {}
    try:
        handle = Path(path).open("r", encoding="utf-8")
    except OSError as error:
        raise TrackedStateProtocolError(f"cannot open tracked-state predictions: {error}") from error
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise TrackedStateProtocolError(f"invalid tracked-state JSONL at line {line_number}") from error
            query_id = record.get("query_id") if isinstance(record, dict) else None
            if query_id not in indexed:
                raise TrackedStateProtocolError(f"unknown tracked-state query ID at line {line_number}: {query_id!r}")
            if query_id in predictions:
                raise TrackedStateProtocolError(f"duplicate tracked-state prediction: {query_id}")
            predictions[query_id] = parse_tracked_state_prediction(record, scope=indexed[query_id])
    if set(predictions) != set(indexed):
        missing = sorted(set(indexed) - set(predictions))
        raise TrackedStateProtocolError(f"tracked-state predictions are incomplete; missing={missing[:3]}")
    return predictions
