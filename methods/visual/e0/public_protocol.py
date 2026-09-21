"""Method-visible data contract for WCM-Grid v2 formal experiments.

This module intentionally accepts only a dataset's ``public`` directory.
Experiment methods must import this module rather than walk a dataset tree on
their own.  In particular it has no label loader and no evaluator dependency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal


FORMAL_SCHEMA_VERSION = "wcm-grid-state-evolution/v2-formal-v1"
K_VALUES = (0, 1, 2, 4, 8, 16)
COMPONENTS = frozenset(("place", "openness", "toggle_state"))
TASKS = frozenset(("state_query", "next_frame"))
RAW_ACTIONS = frozenset(("left", "right", "forward", "pickup", "drop", "toggle"))
SPLITS = frozenset(("train", "validation", "test"))


class PublicProtocolError(ValueError):
    """Raised when a release does not satisfy the frozen method-side contract."""


def _read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise PublicProtocolError(f"cannot read JSON {path}: {error}") from error


def _safe_relative_path(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise PublicProtocolError(f"{field} must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise PublicProtocolError(f"{field} must be a safe relative path: {value!r}")
    return value


def _identifier(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or any(char.isspace() for char in value):
        raise PublicProtocolError(f"{field} must be a non-empty whitespace-free string")
    return value


@dataclass(frozen=True, slots=True)
class PublicFrame:
    """One RGB observation; action produced this frame from its predecessor."""

    frame_index: int
    rgb_path: str
    action: str | None

    @classmethod
    def from_dict(cls, data: object) -> "PublicFrame":
        if not isinstance(data, dict):
            raise PublicProtocolError("public frame must be an object")
        index = data.get("frame_index")
        if not isinstance(index, int) or index < 0:
            raise PublicProtocolError("frame_index must be a non-negative integer")
        action = data.get("action")
        if action is not None and action not in RAW_ACTIONS:
            raise PublicProtocolError(f"unsupported action at frame {index}: {action!r}")
        return cls(index, _safe_relative_path(data.get("rgb_path"), field="rgb_path"), action)

    def to_dict(self) -> dict[str, object]:
        return {"frame_index": self.frame_index, "rgb_path": self.rgb_path, "action": self.action}


@dataclass(frozen=True, slots=True)
class PublicQuery:
    query_id: str
    stream_id: str
    split: str
    task: Literal["state_query", "next_frame"]
    k: int
    prefix_end_frame: int
    entity_id: str
    component: str
    next_action: str | None

    @classmethod
    def from_dict(cls, data: object, *, stream_split: str) -> "PublicQuery":
        if not isinstance(data, dict):
            raise PublicProtocolError("public query must be an object")
        task = data.get("task")
        if task not in TASKS:
            raise PublicProtocolError(f"unsupported query task: {task!r}")
        k = data.get("k")
        if k not in K_VALUES:
            raise PublicProtocolError(f"unsupported K: {k!r}")
        end = data.get("prefix_end_frame")
        if not isinstance(end, int) or end < 0:
            raise PublicProtocolError("prefix_end_frame must be a non-negative integer")
        component = data.get("component")
        if component not in COMPONENTS:
            raise PublicProtocolError(f"unsupported component: {component!r}")
        next_action = data.get("next_action")
        if task == "next_frame":
            if next_action not in RAW_ACTIONS:
                raise PublicProtocolError("next_frame query requires a valid next_action")
        elif next_action is not None:
            raise PublicProtocolError("state_query must not expose next_action")
        return cls(
            query_id=_identifier(data.get("query_id"), field="query_id"),
            stream_id=_identifier(data.get("stream_id"), field="stream_id"),
            split=stream_split,
            task=task,
            k=k,
            prefix_end_frame=end,
            entity_id=_identifier(data.get("entity_id"), field="entity_id"),
            component=component,
            next_action=next_action,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "stream_id": self.stream_id,
            "split": self.split,
            "task": self.task,
            "k": self.k,
            "prefix_end_frame": self.prefix_end_frame,
            "entity_id": self.entity_id,
            "component": self.component,
            "next_action": self.next_action,
        }


@dataclass(frozen=True, slots=True)
class PublicStream:
    stream_id: str
    split: str
    camera: str
    frames: tuple[PublicFrame, ...]
    entity_catalog: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class PublicMethodInput:
    """The entire and only input supplied to one formal method invocation."""

    query: PublicQuery
    camera: str
    entity_catalog: tuple[dict[str, object], ...]
    frames: tuple[PublicFrame, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": FORMAL_SCHEMA_VERSION,
            "query": self.query.to_dict(),
            "camera": self.camera,
            "entity_catalog": list(self.entity_catalog),
            "frames": [frame.to_dict() for frame in self.frames],
        }


@dataclass(frozen=True, slots=True)
class PublicBenchmark:
    """Validated public benchmark index.  It cannot load evaluator labels."""

    public_root: Path
    streams: dict[str, PublicStream]
    queries: dict[str, PublicQuery]

    def method_input(self, query_id: str) -> PublicMethodInput:
        query = self.queries.get(query_id)
        if query is None:
            raise KeyError(f"unknown public query: {query_id}")
        stream = self.streams[query.stream_id]
        return PublicMethodInput(
            query=query,
            camera=stream.camera,
            entity_catalog=stream.entity_catalog,
            frames=stream.frames[: query.prefix_end_frame + 1],
        )

    def queries_for(self, *, split: str | None = None, task: str | None = None) -> tuple[PublicQuery, ...]:
        return tuple(
            query
            for query in self.queries.values()
            if (split is None or query.split == split) and (task is None or query.task == task)
        )


def load_public_benchmark(public_root: str | Path) -> PublicBenchmark:
    """Load and validate only the released method-visible directory."""

    root = Path(public_root).resolve()
    if root.name != "public":
        raise PublicProtocolError("load_public_benchmark requires the dataset public directory exactly")
    streams_doc = _read_json(root / "streams.json")
    queries_doc = _read_json(root / "queries.json")
    catalogs_doc = _read_json(root / "entity_catalogs.json")
    if streams_doc.get("schema_version") != FORMAL_SCHEMA_VERSION:
        raise PublicProtocolError("unexpected streams schema version")
    if queries_doc.get("schema_version") != FORMAL_SCHEMA_VERSION:
        raise PublicProtocolError("unexpected queries schema version")
    stream_records = streams_doc.get("streams")
    query_records = queries_doc.get("queries")
    catalog_records = catalogs_doc.get("streams")
    if not all(isinstance(value, list) for value in (stream_records, query_records, catalog_records)):
        raise PublicProtocolError("streams, queries, and catalogs must be lists")

    catalogs: dict[str, tuple[dict[str, object], ...]] = {}
    for record in catalog_records:
        if not isinstance(record, dict):
            raise PublicProtocolError("entity catalog entry must be an object")
        stream_id = _identifier(record.get("stream_id"), field="catalog.stream_id")
        entities = record.get("entity_catalog")
        if not isinstance(entities, list) or not entities:
            raise PublicProtocolError(f"catalog for {stream_id} must be a non-empty list")
        entity_ids: set[str] = set()
        normalized: list[dict[str, object]] = []
        for entity in entities:
            if not isinstance(entity, dict):
                raise PublicProtocolError(f"catalog entity in {stream_id} must be an object")
            entity_id = _identifier(entity.get("entity_id"), field="entity_id")
            if entity_id in entity_ids:
                raise PublicProtocolError(f"duplicate public entity {entity_id} in {stream_id}")
            entity_ids.add(entity_id)
            normalized.append(dict(entity))
        if stream_id in catalogs:
            raise PublicProtocolError(f"duplicate catalog for stream {stream_id}")
        catalogs[stream_id] = tuple(normalized)

    streams: dict[str, PublicStream] = {}
    for record in stream_records:
        if not isinstance(record, dict):
            raise PublicProtocolError("stream record must be an object")
        stream_id = _identifier(record.get("stream_id"), field="stream_id")
        split = record.get("split")
        if split not in SPLITS:
            raise PublicProtocolError(f"invalid split for {stream_id}: {split!r}")
        if record.get("camera") != "third_person_local_axis_aligned":
            raise PublicProtocolError(f"unexpected camera for {stream_id}")
        frames_path = _safe_relative_path(record.get("frames_path"), field="frames_path")
        frame_doc = _read_json(root / frames_path)
        frame_records = frame_doc.get("frames") if isinstance(frame_doc, dict) else None
        if not isinstance(frame_records, list) or not frame_records:
            raise PublicProtocolError(f"stream {stream_id} has no public frames")
        frames = tuple(PublicFrame.from_dict(frame) for frame in frame_records)
        if tuple(frame.frame_index for frame in frames) != tuple(range(len(frames))):
            raise PublicProtocolError(f"frame indices are not contiguous in {stream_id}")
        if frames[0].action is not None:
            raise PublicProtocolError(f"frame zero action must be null in {stream_id}")
        if record.get("frame_count") != len(frames):
            raise PublicProtocolError(f"declared frame count mismatch in {stream_id}")
        for frame in frames:
            if not (root / frame.rgb_path).is_file():
                raise PublicProtocolError(f"missing public RGB file: {frame.rgb_path}")
        if stream_id not in catalogs:
            raise PublicProtocolError(f"stream {stream_id} has no public entity catalog")
        if stream_id in streams:
            raise PublicProtocolError(f"duplicate public stream: {stream_id}")
        streams[stream_id] = PublicStream(stream_id, split, record["camera"], frames, catalogs[stream_id])

    if set(catalogs) != set(streams):
        raise PublicProtocolError("catalog stream IDs must exactly match public stream IDs")

    queries: dict[str, PublicQuery] = {}
    for record in query_records:
        stream_id = record.get("stream_id") if isinstance(record, dict) else None
        if stream_id not in streams:
            raise PublicProtocolError(f"query refers to unknown stream {stream_id!r}")
        stream = streams[stream_id]
        query = PublicQuery.from_dict(record, stream_split=stream.split)
        if query.prefix_end_frame >= len(stream.frames):
            raise PublicProtocolError(f"query prefix exceeds stream frames: {query.query_id}")
        if query.entity_id not in {entity["entity_id"] for entity in stream.entity_catalog}:
            raise PublicProtocolError(f"query entity absent from public catalog: {query.query_id}")
        if query.query_id in queries:
            raise PublicProtocolError(f"duplicate query ID: {query.query_id}")
        queries[query.query_id] = query
    if not queries:
        raise PublicProtocolError("public benchmark has no queries")
    return PublicBenchmark(root, streams, queries)
