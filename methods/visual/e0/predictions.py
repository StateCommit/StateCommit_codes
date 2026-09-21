"""Strict, method-agnostic prediction records for the formal benchmark."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .public_protocol import PublicProtocolError, PublicQuery


def _safe_relative_png(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise PublicProtocolError("png_path must be a non-empty relative PNG path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.suffix.lower() != ".png":
        raise PublicProtocolError("png_path must be a safe relative PNG path")
    return value


def validate_state_answer(component: str, answer: object) -> str | dict[str, str]:
    if component == "openness":
        if answer not in {"open", "closed"}:
            raise PublicProtocolError("openness answer must be 'open' or 'closed'")
        return answer
    if component == "toggle_state":
        if answer not in {"on", "off"}:
            raise PublicProtocolError("toggle_state answer must be 'on' or 'off'")
        return answer
    if component == "place":
        if not isinstance(answer, dict) or set(answer) != {"kind", "target"}:
            raise PublicProtocolError("place answer must be exactly {'kind', 'target'}")
        if answer.get("kind") != "in_zone" or not isinstance(answer.get("target"), str):
            raise PublicProtocolError("place answer must be {'kind':'in_zone','target':'zone_i'}")
        return {"kind": "in_zone", "target": answer["target"]}
    raise PublicProtocolError(f"unsupported component: {component!r}")


@dataclass(frozen=True, slots=True)
class StatePrediction:
    query_id: str
    answer: str | dict[str, str]


@dataclass(frozen=True, slots=True)
class NextFramePrediction:
    query_id: str
    png_path: str


def parse_prediction_record(record: object, *, query: PublicQuery) -> StatePrediction | NextFramePrediction:
    if not isinstance(record, dict) or set(record) - {"query_id", "task", "answer", "png_path"}:
        raise PublicProtocolError("prediction record has unsupported fields")
    if record.get("query_id") != query.query_id or record.get("task") != query.task:
        raise PublicProtocolError(f"prediction identity/task mismatch for {query.query_id}")
    if query.task == "state_query":
        if set(record) != {"query_id", "task", "answer"}:
            raise PublicProtocolError("state prediction requires exactly query_id, task, answer")
        return StatePrediction(query.query_id, validate_state_answer(query.component, record.get("answer")))
    if set(record) != {"query_id", "task", "png_path"}:
        raise PublicProtocolError("next-frame prediction requires exactly query_id, task, png_path")
    return NextFramePrediction(query.query_id, _safe_relative_png(record.get("png_path")))


def read_prediction_jsonl(path: str | Path, queries: dict[str, PublicQuery]) -> dict[str, StatePrediction | NextFramePrediction]:
    """Read one strict record per known query; duplicate or stray IDs fail closed."""

    predictions: dict[str, StatePrediction | NextFramePrediction] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record: Any = json.loads(line)
            except json.JSONDecodeError as error:
                raise PublicProtocolError(f"invalid JSONL at line {line_number}: {error}") from error
            query_id = record.get("query_id") if isinstance(record, dict) else None
            if query_id not in queries:
                raise PublicProtocolError(f"unknown query ID at line {line_number}: {query_id!r}")
            if query_id in predictions:
                raise PublicProtocolError(f"duplicate prediction for {query_id}")
            predictions[query_id] = parse_prediction_record(record, query=queries[query_id])
    return predictions
