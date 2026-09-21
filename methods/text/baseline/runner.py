"""Resumable, online-by-episode execution for every E1 memory method."""

from __future__ import annotations

import json
import inspect
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping

from e1core.protocol import QueryExample, canonical_json, load_examples, read_json, sha256_file
from e1core.transition_spec import bind_dataset_transition_spec, spec_provenance

from .base import MemoryMethod, call_usage, public_scope


RUNNER_VERSION = "statereturn-text-online-runner/v1"


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    payload = "".join(canonical_json(row) + "\n" for row in rows)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _manifest(dataset_root: Path, *, method: MemoryMethod, split: str, representation: str, options: Mapping[str, Any]) -> dict[str, Any]:
    release = dataset_root / "metadata" / "release_manifest.json"
    source = inspect.getsourcefile(type(method))
    if source is None:
        raise RuntimeError(f"cannot bind implementation source for {type(method).__name__}")
    return {
        "runner_version": RUNNER_VERSION,
        "method": method.name,
        "split": split,
        "representation": representation,
        "dataset": {
            "root": str(dataset_root.resolve()),
            "release_manifest_sha256": sha256_file(release),
            "data_sha256": sha256_file(dataset_root / "data" / representation / f"{split}.jsonl"),
        },
        "options": dict(options),
        "controller_implementation": {
            "class": f"{type(method).__module__}.{type(method).__qualname__}",
            "source_sha256": sha256_file(source),
        },
        "public_transition_spec": spec_provenance(representation),
        "dataset_transition_spec": bind_dataset_transition_spec(dataset_root, representation),
        "private_data_access_during_run": False,
    }


def _reset_gpu_peak(device: Any) -> None:
    """Reset peak accounting after a method backend is initialized, if available."""

    try:
        import torch

        if isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
    except (ImportError, RuntimeError, ValueError):
        return


def _gpu_usage(device: Any) -> dict[str, Any]:
    try:
        import torch

        if isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available():
            return {
                "device": device,
                "gpu_memory_allocated_bytes": int(torch.cuda.memory_allocated(device)),
                "peak_gpu_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            }
    except (ImportError, RuntimeError, ValueError):
        pass
    return {"device": device, "gpu_memory_allocated_bytes": None, "peak_gpu_memory_allocated_bytes": None}


def _progress(*, completed: int, total: int, started: float) -> dict[str, Any]:
    elapsed = time.monotonic() - started
    seconds_each = elapsed / completed if completed else None
    return {
        "status": "running" if completed < total else "completed",
        "completed_episodes": completed,
        "total_episodes": total,
        "remaining_episodes": total - completed,
        "elapsed_seconds": round(elapsed, 3),
        "estimated_remaining_seconds": None if seconds_each is None else round((total - completed) * seconds_each, 3),
    }


def _group(examples: list[QueryExample]) -> dict[str, list[QueryExample]]:
    result: dict[str, list[QueryExample]] = defaultdict(list)
    for example in examples:
        result[example.episode_id].append(example)
    return {episode: sorted(rows, key=lambda item: item.checkpoint_k) for episode, rows in result.items()}


def _read_episode(path: Path) -> list[Mapping[str, Any]]:
    value = read_json(path)
    rows = value.get("predictions")
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise ValueError(f"malformed completed episode cache: {path}")
    return rows


def run_online(
    *, dataset_root: str | Path, split: str, output_root: str | Path,
    method_factory: Callable[[], MemoryMethod], options: Mapping[str, Any], representation: str = "structured",
    limit_episodes: int | None = None,
) -> dict[str, Any]:
    """Process each episode once, persisting only complete episode checkpoints."""

    dataset, output = Path(dataset_root), Path(output_root)
    examples = load_examples(dataset, split=split, representation=representation)
    grouped = _group(examples)
    if limit_episodes is not None:
        if limit_episodes <= 0:
            raise ValueError("limit_episodes must be positive")
        selected_ids = sorted(grouped)[:limit_episodes]
        grouped = {episode_id: grouped[episode_id] for episode_id in selected_ids}
    probe = method_factory()
    manifest = _manifest(
        dataset,
        method=probe,
        split=split, representation=representation,
        options={**dict(options), "limited_smoke_selection": limit_episodes is not None},
    )
    manifest_path = output / "run_manifest.json"
    if manifest_path.exists() and read_json(manifest_path) != manifest:
        raise ValueError("resume refused: run manifest differs")
    _atomic_json(manifest_path, manifest)
    device = options.get("device")
    _reset_gpu_peak(device)
    cache = output / "episodes"
    started = time.monotonic()
    completed = 0
    for episode_id, checkpoints in sorted(grouped.items()):
        episode_path = cache / f"{episode_id}.json"
        expected_ids = [item.query_id for item in checkpoints]
        if episode_path.is_file():
            rows = _read_episode(episode_path)
            if [row.get("query_id") for row in rows] != expected_ids:
                raise ValueError(f"cached episode has incompatible query IDs: {episode_id}")
            completed += 1
            continue
        method = method_factory()
        method.reset()
        prior_events: list[str] = []
        rows: list[Mapping[str, Any]] = []
        for example in checkpoints:
            current_ids = [event.event_id for event in example.history]
            if current_ids[: len(prior_events)] != prior_events:
                raise ValueError(f"non-prefix episode history: {episode_id}")
            event_calls = []
            event_update_seconds = 0.0
            for event in example.history[len(prior_events) :]:
                event_started = time.monotonic()
                event_calls.extend(method.step(event))
                event_update_seconds += time.monotonic() - event_started
            prior_events = current_ids
            scope = public_scope(example.history, query_entity_id=example.target_entity_id)
            answer_started = time.monotonic()
            answer, answer_calls = method.answer(example)
            answer_seconds = time.monotonic() - answer_started
            dump_started = time.monotonic()
            state, dump_calls = method.dump(scope)
            dump_seconds = time.monotonic() - dump_started
            calls = [*event_calls, *answer_calls, *dump_calls]
            usage = call_usage(calls)
            usage.update(
                {
                    "event_update_seconds": round(event_update_seconds, 6),
                    "query_seconds": round(answer_seconds, 6),
                    "state_dump_seconds": round(dump_seconds, 6),
                    "total_online_seconds": round(event_update_seconds + answer_seconds + dump_seconds, 6),
                    "method_compute": dict(method.compute_usage()),
                }
            )
            rows.append(
                {
                    "query_id": example.query_id,
                    "episode_id": example.episode_id,
                    "checkpoint_k": example.checkpoint_k,
                    "target_component": example.target_component,
                    "public_probe_scope": list(scope),
                    "public_query_fingerprint": example.public_fingerprint(),
                    "prediction": answer,
                    "predicted_tracked_state": state,
                    "agent_calls": [call.to_dict() for call in calls],
                    "agent_usage": usage,
                    "controller_trace": method.trace(),
                }
            )
        _atomic_json(episode_path, {"episode_id": episode_id, "predictions": rows})
        completed += 1
        _atomic_json(output / "progress.json", _progress(completed=completed, total=len(grouped), started=started))
    all_rows = [row for episode in sorted(grouped) for row in _read_episode(cache / f"{episode}.json")]
    _atomic_jsonl(output / "predictions.jsonl", all_rows)
    result = _progress(completed=len(grouped), total=len(grouped), started=started)
    result.update(
        {
            "prediction_path": str(output / "predictions.jsonl"),
            "query_count": len(all_rows),
            "method": probe.name,
            "run_resource_usage": _gpu_usage(device),
        }
    )
    _atomic_json(output / "progress.json", result)
    return result
