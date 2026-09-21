"""Public I/O contract for State-Consistent Next-Frame Prediction.

This module prepares only causal renderer inputs and validates produced PNG
records.  It intentionally has no factual-target, counterfactual-target,
mask, simulator-state, or scoring import.
"""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any

from .visual import VisualPublicError, _canonical, load_visual_benchmark


def build_public_next_frame_jobs(dataset_root: str | Path, *, split: str, output_path: str | Path) -> dict[str, object]:
    """Write one renderer job per causal next-frame query without targets."""

    benchmark = load_visual_benchmark(dataset_root)
    if split not in {"train", "validation", "test"}:
        raise VisualPublicError("unsupported split")
    jobs: list[dict[str, object]] = []
    for query in sorted(benchmark.queries.values(), key=lambda item: item.query_id):
        if query.split != split or query.task != "next_frame":
            continue
        stream = benchmark.streams[query.stream_id]
        frames = stream.frames[:query.prefix_end_frame + 1]
        jobs.append({
            "query_id": query.query_id,
            "task": "next_frame",
            "stream_id": query.stream_id,
            "k": query.k,
            "entity_id": query.entity_id,
            "component": query.component,
            "next_action": query.next_action,
            "current_rgb_path": frames[-1].rgb_path,
            "causal_prefix": [{"frame_index": frame.frame_index, "rgb_path": frame.rgb_path, "action": frame.action} for frame in frames],
            "entity_catalog": list(stream.entity_catalog),
            "method_never_receives": ["factual future RGB", "counterfactual future RGB", "source mask", "state answer", "simulator metadata"],
        })
    expected = sum(query.split == split and query.task == "next_frame" for query in benchmark.queries.values())
    if not expected or len(jobs) != expected:
        raise VisualPublicError("selected split has incomplete or unavailable public next-frame queries")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(_canonical(job) + "\n" for job in jobs), encoding="utf-8")
    return {"status": "public_jobs_complete", "split": split, "job_count": len(jobs), "output_path": str(destination)}


def validate_next_frame_predictions(predictions_path: str | Path, *, jobs_path: str | Path, output_root: str | Path) -> dict[str, object]:
    """Validate a renderer's public prediction file; never score it.

    Every prediction must be ``{query_id, task, png_path}``, cover exactly the
    jobs supplied to that renderer, and resolve to a PNG beneath ``output_root``.
    """

    expected: set[str] = set()
    for line in Path(jobs_path).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if not isinstance(row, dict) or not isinstance(row.get("query_id"), str):
            raise VisualPublicError("public next-frame job file is malformed")
        expected.add(row["query_id"])
    root = Path(output_root).resolve()
    seen: set[str] = set()
    for line_number, line in enumerate(Path(predictions_path).read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise VisualPublicError(f"invalid prediction JSONL at line {line_number}") from exc
        if not isinstance(row, dict) or set(row) != {"query_id", "task", "png_path"} or row.get("task") != "next_frame":
            raise VisualPublicError("next-frame prediction requires exactly query_id/task/png_path")
        query_id, relative = row.get("query_id"), row.get("png_path")
        if not isinstance(query_id, str) or query_id not in expected or query_id in seen:
            raise VisualPublicError("unknown or duplicate next-frame prediction")
        if not isinstance(relative, str):
            raise VisualPublicError("png_path must be a relative string")
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts or path.suffix.lower() != ".png":
            raise VisualPublicError("png_path is unsafe or not a PNG")
        target = (root / path).resolve()
        if root not in target.parents or not target.is_file():
            raise VisualPublicError("prediction PNG is missing or escapes output_root")
        seen.add(query_id)
    if seen != expected:
        raise VisualPublicError(f"prediction coverage mismatch: expected {len(expected)}, got {len(seen)}")
    return {"status": "public_prediction_format_valid", "prediction_count": len(seen), "evaluator_assets_opened": False}
