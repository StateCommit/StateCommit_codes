"""Causal public-input runner for E1 Visual WCM state-return queries."""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from e0.public_protocol import PublicBenchmark, PublicQuery
from .fast_path_agents import (
    DeterministicWorldCodeProgrammer,
    GroundedPatchAuditor,
    VisualGrounder,
    WorldCodeProgrammer,
)
from .tracked_state_protocol import PublicTrackedStateScopes, TRACKED_STATE_TASK
from .visual_wcm_runtime import ActiveStateRuntime, RuntimeViolation, WorldCodeSyntaxError, parse_state_patch_code


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _state_queries(benchmark: PublicBenchmark, *, split: str) -> dict[str, list[PublicQuery]]:
    grouped: dict[str, list[PublicQuery]] = defaultdict(list)
    for query in benchmark.queries_for(split=split, task="state_query"):
        grouped[query.stream_id].append(query)
    return {stream_id: sorted(queries, key=lambda query: (query.prefix_end_frame, query.query_id)) for stream_id, queries in grouped.items()}


def _typed_public_fallback(*, runtime: ActiveStateRuntime, entity_id: str, component: str) -> str | dict[str, str]:
    """Return a syntactically valid public placeholder for an untracked slot.

    The sidecar evaluator receives ``untracked_slots`` and always scores this
    slot as incorrect.  A typed fallback exists solely so an abstaining visual
    Grounder cannot corrupt JSONL alignment or hide coverage failure.
    """

    if component == "openness":
        return "closed"
    if component == "toggle_state":
        return "off"
    if component == "place":
        zones = runtime.binding_store.zone_ids()
        if not zones:
            raise RuntimeViolation("public catalog has no zone for place fallback")
        return {"kind": "in_zone", "target": zones[0]}
    raise RuntimeViolation(f"cannot build fallback for unsupported component {component!r}")


def _answer_or_untracked(
    *, runtime: ActiveStateRuntime, entity_id: str, component: str
) -> tuple[str | dict[str, str], bool]:
    try:
        return runtime.answer(entity_id=entity_id, component=component), False
    except RuntimeViolation as error:
        if "untracked" not in str(error):
            raise
        return _typed_public_fallback(runtime=runtime, entity_id=entity_id, component=component), True


def _dump_or_untracked(
    *, runtime: ActiveStateRuntime, slots: tuple[str, ...]
) -> tuple[dict[str, str | dict[str, str]], list[str]]:
    state: dict[str, str | dict[str, str]] = {}
    missing: list[str] = []
    for slot in slots:
        entity_id, component = slot.split("|", 1)
        value, untracked = _answer_or_untracked(runtime=runtime, entity_id=entity_id, component=component)
        state[slot] = value
        if untracked:
            missing.append(slot)
    return state, missing


def run_visual_wcm_state_queries(
    *,
    benchmark: PublicBenchmark,
    split: str,
    grounder: VisualGrounder,
    output_root: str | Path,
    tracked_state_scopes: PublicTrackedStateScopes | None = None,
    programmer: WorldCodeProgrammer | None = None,
    auditor: GroundedPatchAuditor | None = None,
) -> dict[str, object]:
    """Replay public streams through the Visual WCM executable Fast Path.

    Grounder, Coding Programmer, Auditor, and Runtime are deliberately
    separate: no component may bypass the literal-only patch parser or the
    Runtime's version/type/expected-old checks.
    """

    root = Path(output_root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"output root is not empty: {root}")
    root.mkdir(parents=True, exist_ok=False)
    programmer = programmer or DeterministicWorldCodeProgrammer()
    auditor = auditor or GroundedPatchAuditor()
    grouped = _state_queries(benchmark, split=split)
    expected_query_count = sum(len(queries) for queries in grouped.values())
    prediction_path = root / "state_predictions.jsonl"
    tracked_prediction_path = root / "tracked_state_predictions.jsonl"
    traces_path = root / "stream_runtime_traces.jsonl"
    action_counts: Counter[str] = Counter()
    committed_components: Counter[str] = Counter()
    programmer_call_count = 0
    auditor_counts: Counter[str] = Counter()
    code_parse_failure_count = 0
    grounder_no_fact_count = 0
    untracked_target_query_count = 0
    untracked_slot_occurrence_count = 0
    completed_queries = 0
    for stream_ordinal, stream_id in enumerate(sorted(grouped), start=1):
        stream, queries = benchmark.streams[stream_id], grouped[stream_id]
        checkpoints: dict[int, list[PublicQuery]] = defaultdict(list)
        for query in queries:
            checkpoints[query.prefix_end_frame].append(query)
        runtime = ActiveStateRuntime(entity_catalog=stream.entity_catalog)
        for frame in stream.frames:
            if frame.frame_index > 0 and frame.action in {"drop", "toggle"}:
                before = stream.frames[frame.frame_index - 1]
                fact = grounder.ground(
                    before_image=benchmark.public_root / before.rgb_path,
                    after_image=benchmark.public_root / frame.rgb_path,
                    action=frame.action,
                    entity_catalog=stream.entity_catalog,
                )
                event_id = f"{stream_id}:frame_{frame.frame_index:06d}"
                action_counts[frame.action] += 1
                if fact is None:
                    grounder_no_fact_count += 1
                else:
                    raw_code = programmer.propose(fact=fact, runtime=runtime, event_id=event_id)
                    programmer_call_count += 1
                    try:
                        patch = parse_state_patch_code(raw_code)
                    except WorldCodeSyntaxError as error:
                        code_parse_failure_count += 1
                        auditor_counts["reject:invalid_world_code"] += 1
                        raise RuntimeViolation(f"Coding Programmer emitted invalid World Code for {event_id}: {error}") from error
                    audit = auditor.audit(fact=fact, patch=patch, runtime=runtime, event_id=event_id)
                    auditor_counts[audit.verdict if audit.reason is None else f"{audit.verdict}:{audit.reason}"] += 1
                    if audit.verdict != "support":
                        raise RuntimeViolation(f"Auditor rejected world code for {event_id}: {audit.reason}")
                    runtime.commit(patch)
                    committed_components[fact.component] += 1
            for query in checkpoints.get(frame.frame_index, []):
                answer, target_untracked = _answer_or_untracked(
                    runtime=runtime, entity_id=query.entity_id, component=query.component
                )
                untracked_target_query_count += int(target_untracked)
                _append_jsonl(prediction_path, {"query_id": query.query_id, "task": "state_query", "answer": answer})
                if tracked_state_scopes is not None:
                    scope = tracked_state_scopes.for_query(query)
                    dump, missing = _dump_or_untracked(runtime=runtime, slots=scope.tracked_slots)
                    untracked_slot_occurrence_count += len(missing)
                    _append_jsonl(
                        tracked_prediction_path,
                        {
                            "query_id": query.query_id,
                            "task": TRACKED_STATE_TASK,
                            "state": dump,
                            "untracked_slots": missing,
                        },
                    )
                completed_queries += 1
        _append_jsonl(
            traces_path,
            {
                "stream_id": stream_id,
                "split": stream.split,
                "last_public_frame": stream.frames[-1].frame_index,
                "runtime": runtime.snapshot(),
                "committed_components": dict(sorted(Counter(receipt.component for receipt in runtime.transaction_log).items())),
                "fast_path_agents": {
                    "grounder": getattr(grounder, "name", type(grounder).__name__),
                    "programmer": programmer.name,
                    "auditor": auditor.name,
                },
            },
        )
        _atomic_json(
            root / "progress.json",
            {
                "status": "running",
                "completed_streams": stream_ordinal,
                "total_streams": len(grouped),
                "completed_state_queries": completed_queries,
                "total_state_queries": expected_query_count,
            },
        )
    if completed_queries != expected_query_count:
        raise RuntimeError("Visual WCM prediction coverage is incomplete")
    report = {
        "stage": "E1_visual_wcm_state_return_public_replay",
        "status": "public_predictions_complete",
        "split": split,
        "stream_count": len(grouped),
        "state_query_count": completed_queries,
        "grounder_invocation_count": sum(action_counts.values()),
        "grounder_invocations_by_action": dict(sorted(action_counts.items())),
        "grounder_no_fact_count": grounder_no_fact_count,
        "grounder_invalid_fact_count": int(getattr(grounder, "invalid_fact_count", 0)),
        "programmer_call_count": programmer_call_count,
        "programmer": programmer.name,
        "auditor": auditor.name,
        "auditor_verdict_counts": dict(sorted(auditor_counts.items())),
        "world_code_parse_failure_count": code_parse_failure_count,
        "untracked_target_query_count": untracked_target_query_count,
        "untracked_slot_occurrence_count": untracked_slot_occurrence_count,
        "runtime_commits_by_component": dict(sorted(committed_components.items())),
        "fast_path": "Visual Grounder -> Coding Programmer (restricted set_state) -> Grounded Patch Auditor -> versioned Runtime atomic commit.",
        "causality": "Each query answer is read at its prefix_end_frame before later public frames are processed.",
        "method_inputs": ["complete public RGB frames", "raw public actions", "public entity catalog", "public query"],
        "method_never_reads": ["evaluator labels", "oracle state", "plans", "action success", "masks", "reveal RGB"],
        "use_public_change_crop": False,
        "tracked_state_prediction_file": (
            "tracked_state_predictions.jsonl" if tracked_state_scopes is not None else None
        ),
    }
    _atomic_json(root / "public_run_report.json", report)
    _atomic_json(root / "progress.json", {"status": "public_predictions_complete", "completed_state_queries": completed_queries, "total_state_queries": expected_query_count})
    return report
