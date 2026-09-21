"""P1: causally compile Visual WCM state and visual bindings for E2.

Only the P0 public index and frozen G6 public RGB/action streams are read.
For every query, the artifact stores the Runtime-produced Active State plus
the latest public after-event frame bound to the queried entity/slot.  It
never opens target RGB, evaluator masks, source plans, or simulator state.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

from e0.public_protocol import PublicBenchmark, PublicQuery, load_public_benchmark
from e0_5.learned_grounder import LearnedFullFrameGrounder
from e1.fast_path_agents import GroundedPatchAuditor, QwenCoderWorldCodeProgrammer
from e1.visual_wcm_runtime import ActiveStateRuntime, WorldCodeSyntaxError, parse_state_patch_code


P1_SCHEMA_VERSION = "statereturn-visual-p1-condition/v1"
SPLITS = ("train", "validation", "test")
UNTRACKED = "__WCM_UNTRACKED__"


class WCMConditionError(RuntimeError):
    """A public P0 input or a causal WCM condition artifact is malformed."""


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> str:
    data = b"".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        for row in rows
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise WCMConditionError(f"malformed JSONL at {path}:{number}") from error
        if not isinstance(value, dict):
            raise WCMConditionError(f"P0 row at {path}:{number} is not an object")
        rows.append(value)
    return rows


def _validated_p0_rows(*, p0_root: Path, benchmark: PublicBenchmark, split: str) -> dict[str, dict[str, object]]:
    path = p0_root / "public" / f"{split}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_fields = {
        "schema_version", "sample_id", "query_id", "stream_id", "split", "k", "prefix_end_frame",
        "history_frame_count", "current_rgb_path", "next_action", "entity_id", "component",
    }
    rows: dict[str, dict[str, object]] = {}
    for row in _read_jsonl(path):
        if set(row) != expected_fields or row.get("schema_version") != "wcm-grid-state-evolution/v2-e2-next-frame-p0/v1":
            raise WCMConditionError("P0 public row schema mismatch")
        query_id = row.get("query_id")
        if not isinstance(query_id, str) or query_id in rows:
            raise WCMConditionError("P0 query ID is missing or duplicated")
        query = benchmark.queries.get(query_id)
        if query is None or query.task != "next_frame" or query.split != split:
            raise WCMConditionError("P0 row refers to an invalid public next-frame query")
        expected = {
            "sample_id": query_id, "stream_id": query.stream_id, "split": split, "k": query.k,
            "prefix_end_frame": query.prefix_end_frame, "history_frame_count": query.prefix_end_frame + 1,
            "next_action": query.next_action, "entity_id": query.entity_id, "component": query.component,
            "current_rgb_path": benchmark.streams[query.stream_id].frames[query.prefix_end_frame].rgb_path,
        }
        if any(row[field] != value for field, value in expected.items()):
            raise WCMConditionError("P0 row disagrees with frozen public G6 input")
        rows[query_id] = row
    expected_ids = {query.query_id for query in benchmark.queries.values() if query.task == "next_frame" and query.split == split}
    if set(rows) != expected_ids:
        raise WCMConditionError("P0 rows do not exactly cover their public split")
    return rows


def _binding_record(*, frame, event_id: str, entity_id: str, component: str) -> dict[str, object]:
    return {
        "entity_id": entity_id,
        "component": component,
        "event_id": event_id,
        "after_frame_index": frame.frame_index,
        "after_rgb_path": frame.rgb_path,
    }


def _condition_row(
    *, query: PublicQuery, p0_row: dict[str, object], runtime: ActiveStateRuntime, bindings: dict[str, dict[str, object]]
) -> dict[str, object]:
    slot = f"{query.entity_id}|{query.component}"
    state = runtime.active_state_snapshot()
    target_value = state.get(slot, UNTRACKED)
    binding = bindings.get(slot)
    if (target_value == UNTRACKED) != (binding is None):
        raise WCMConditionError("Runtime state and Visual Binding existence disagree")
    return {
        "schema_version": P1_SCHEMA_VERSION,
        "sample_id": query.query_id,
        "query_id": query.query_id,
        "stream_id": query.stream_id,
        "split": query.split,
        "k": query.k,
        "prefix_end_frame": query.prefix_end_frame,
        "current_rgb_path": p0_row["current_rgb_path"],
        "next_action": query.next_action,
        "query_slot": slot,
        "active_state": state,
        "compiled_query_state": {"entity_id": query.entity_id, "component": query.component, "value": target_value},
        "visual_binding": binding,
        "is_query_slot_untracked": target_value == UNTRACKED,
    }


def _process_stream(
    *, benchmark: PublicBenchmark, stream_id: str, queries: list[PublicQuery], p0_rows: dict[str, dict[str, object]],
    grounder: LearnedFullFrameGrounder, programmer: QwenCoderWorldCodeProgrammer, auditor: GroundedPatchAuditor,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    stream = benchmark.streams[stream_id]
    checkpoints: dict[int, list[PublicQuery]] = defaultdict(list)
    for query in queries:
        checkpoints[query.prefix_end_frame].append(query)
    runtime = ActiveStateRuntime(entity_catalog=stream.entity_catalog)
    bindings: dict[str, dict[str, object]] = {}
    telemetry: Counter[str] = Counter()
    rows: list[dict[str, object]] = []
    for frame in stream.frames:
        if frame.frame_index > 0 and frame.action in {"drop", "toggle"}:
            before = stream.frames[frame.frame_index - 1]
            event_id = f"{stream_id}:frame_{frame.frame_index:06d}"
            telemetry[f"grounder:{frame.action}"] += 1
            fact = grounder.ground(
                before_image=benchmark.public_root / before.rgb_path,
                after_image=benchmark.public_root / frame.rgb_path,
                action=frame.action,
                entity_catalog=stream.entity_catalog,
            )
            raw_code = programmer.propose(fact=fact, runtime=runtime, event_id=event_id)
            telemetry["programmer_calls"] += 1
            try:
                patch = parse_state_patch_code(raw_code)
            except WorldCodeSyntaxError:
                telemetry["auditor:reject_invalid_world_code"] += 1
                continue
            receipt = auditor.audit(fact=fact, patch=patch, runtime=runtime, event_id=event_id)
            telemetry[f"auditor:{receipt.verdict}" if receipt.reason is None else f"auditor:{receipt.verdict}:{receipt.reason}"] += 1
            if receipt.verdict != "support":
                continue
            runtime.commit(patch)
            slot = f"{fact.entity_id}|{fact.component}"
            bindings[slot] = _binding_record(
                frame=frame, event_id=event_id, entity_id=fact.entity_id, component=fact.component
            )
            telemetry[f"runtime_commit:{fact.component}"] += 1
        for query in checkpoints.get(frame.frame_index, []):
            row = _condition_row(query=query, p0_row=p0_rows[query.query_id], runtime=runtime, bindings=bindings)
            rows.append(row)
            telemetry["query_untracked"] += int(bool(row["is_query_slot_untracked"]))
    if len(rows) != len(queries):
        raise WCMConditionError(f"incomplete query condition coverage for {stream_id}")
    return sorted(rows, key=lambda row: str(row["query_id"])), {
        "stream_id": stream_id,
        "query_count": len(rows),
        "telemetry": dict(sorted(telemetry.items())),
        "runtime_version": runtime.version,
        "final_active_state_slot_count": len(runtime.active_state_snapshot()),
    }


def build_wcm_conditions(
    *, dataset_root: str | Path, p0_root: str | Path, output_root: str | Path, checkpoint_path: str | Path,
    image_size: int, coder_model_path: str | Path, device: str, splits: Iterable[str], resume: bool = False,
    limit_streams: int | None = None,
) -> dict[str, object]:
    selected = tuple(splits)
    if not selected or any(split not in SPLITS for split in selected) or len(set(selected)) != len(selected):
        raise WCMConditionError(f"invalid split selection: {selected!r}")
    if limit_streams is not None and limit_streams <= 0:
        raise WCMConditionError("limit_streams must be positive when provided")
    dataset, p0, root = Path(dataset_root).resolve(), Path(p0_root).resolve(), Path(output_root).resolve()
    if root.exists() and any(root.iterdir()) and not resume:
        raise FileExistsError(f"condition output exists and is non-empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    benchmark = load_public_benchmark(dataset / "public")
    rows_by_split = {split: _validated_p0_rows(p0_root=p0, benchmark=benchmark, split=split) for split in selected}
    grounder = LearnedFullFrameGrounder(checkpoint_path=checkpoint_path, device=device, image_size=image_size)
    programmer = QwenCoderWorldCodeProgrammer(model_path=coder_model_path, max_new_tokens=96, device=device)
    auditor = GroundedPatchAuditor()
    split_outputs: dict[str, dict[str, object]] = {}
    for split in selected:
        grouped: dict[str, list[PublicQuery]] = defaultdict(list)
        for query_id in rows_by_split[split]:
            query = benchmark.queries[query_id]
            grouped[query.stream_id].append(query)
        stream_ids = sorted(grouped)
        if limit_streams is not None:
            stream_ids = stream_ids[:limit_streams]
        selected_query_count = sum(len(grouped[stream_id]) for stream_id in stream_ids)
        stream_rows: list[dict[str, object]] = []
        total_streams = len(stream_ids)
        for ordinal, stream_id in enumerate(stream_ids, start=1):
            per_stream = root / "streams" / split / f"{stream_id}.jsonl"
            trace_path = root / "streams" / split / f"{stream_id}.trace.json"
            if resume and per_stream.is_file() and trace_path.is_file():
                rows = _read_jsonl(per_stream)
                if {row.get("query_id") for row in rows} != {query.query_id for query in grouped[stream_id]}:
                    raise WCMConditionError(f"resume condition file has incorrect coverage: {per_stream}")
            else:
                rows, trace = _process_stream(
                    benchmark=benchmark, stream_id=stream_id, queries=grouped[stream_id], p0_rows=rows_by_split[split],
                    grounder=grounder, programmer=programmer, auditor=auditor,
                )
                _atomic_jsonl(per_stream, rows)
                _atomic_json(trace_path, trace)
            stream_rows.extend(rows)
            _atomic_json(root / "progress.json", {
                "status": "running", "split": split, "completed_streams": ordinal, "total_streams": total_streams,
                "completed_query_conditions": len(stream_rows), "resume": resume,
            })
        if len(stream_rows) != selected_query_count:
            raise WCMConditionError(f"P1 output count mismatch for {split}")
        output_path = root / "conditions" / f"{split}.jsonl"
        split_outputs[split] = {
            "query_count": len(stream_rows),
            "selected_stream_count": total_streams,
            "available_stream_count": len(grouped),
            "condition_sha256": _atomic_jsonl(output_path, sorted(stream_rows, key=lambda row: str(row["query_id"]))),
            "p0_public_sha256": _sha256_file(p0 / "public" / f"{split}.jsonl"),
        }
    manifest = {
        "schema_version": P1_SCHEMA_VERSION,
        "stage": "E2 P1: causal Visual WCM active-state and visual-binding compilation",
        "selected_splits": list(selected),
        "limit_streams": limit_streams,
        "grounder": {"checkpoint": str(Path(checkpoint_path).resolve()), "image_size": image_size, "device": device},
        "programmer": {"model_path": str(Path(coder_model_path).resolve()), "max_new_tokens": 96, "device": device},
        "auditor": auditor.name,
        "runtime": "visual-wcm-runtime/e1-v2-fast-path",
        "condition_fields": ["active_state", "compiled_query_state", "visual_binding"],
        "causal_binding_rule": "A binding is the after-RGB frame of the latest Auditor-supported Runtime commit for the queried entity|component before prefix_end_frame.",
        "method_never_reads": ["next-frame target RGB", "state labels", "Oracle state", "counterfactual image", "evaluator masks", "source plans"],
        "outputs": split_outputs,
    }
    manifest["manifest_sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    _atomic_json(root / "manifest.json", manifest)
    _atomic_json(root / "progress.json", {"status": "passed", "selected_splits": list(selected), "outputs": split_outputs})
    return manifest
