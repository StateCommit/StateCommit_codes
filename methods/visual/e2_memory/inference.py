"""Public-only prediction runner for frozen visual memory baselines."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader

from e0.public_protocol import PublicBenchmark
from e1.tracked_state_protocol import (
    PublicTrackedStateScopes,
    TRACKED_STATE_TASK,
    load_public_tracked_state_scopes,
    parse_slot_key,
)

from .features import load_feature_cache, sha256_file
from .models import MemoryModelConfig, MemoryStatePredictor
from .protocol import (
    COMPONENT_INDEX,
    MemoryDataset,
    MemoryExample,
    collate_examples,
    decode_slot_value,
    decode_value,
    query_subject_index,
    query_value_mask,
    restrict_logits,
    slot_binding,
)


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


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("metadata must be a JSON object")
    return data


def build_public_examples(*, benchmark: PublicBenchmark, split: str, feature_root: str | Path, grounder_checkpoint: str | Path, image_size: int) -> list[MemoryExample]:
    """Build unlabeled query examples; target is a dummy tensor never consumed."""

    features = load_feature_cache(root=feature_root, benchmark=benchmark, split=split, checkpoint_path=grounder_checkpoint, image_size=image_size)
    examples: list[MemoryExample] = []
    for query in sorted(benchmark.queries_for(split=split, task="state_query"), key=lambda query: query.query_id):
        stream_features, actions = features[query.stream_id]
        examples.append(MemoryExample(
            query_id=query.query_id, stream_id=query.stream_id, slot=f"{query.entity_id}|{query.component}", features=stream_features[:query.prefix_end_frame].contiguous(),
            actions=actions[:query.prefix_end_frame].contiguous(), subject=query_subject_index(query=query, benchmark=benchmark),
            component=COMPONENT_INDEX[query.component], allowed_mask=query_value_mask(query=query, benchmark=benchmark), target=0,
        ))
    return examples


def build_public_tracked_examples(
    *, benchmark: PublicBenchmark, split: str, feature_root: str | Path, grounder_checkpoint: str | Path,
    image_size: int, tracked_state_sidecar: str | Path
) -> tuple[list[MemoryExample], PublicTrackedStateScopes]:
    """Build public-only decoder calls for every slot declared by E1.1."""

    features = load_feature_cache(
        root=feature_root, benchmark=benchmark, split=split, checkpoint_path=grounder_checkpoint, image_size=image_size
    )
    scopes = load_public_tracked_state_scopes(Path(tracked_state_sidecar) / "public", benchmark)
    examples: list[MemoryExample] = []
    for scope in sorted(scopes.for_split(split), key=lambda item: item.query_id):
        stream_features, actions = features[scope.stream_id]
        if scope.prefix_end_frame <= 0 or scope.prefix_end_frame > stream_features.shape[0]:
            raise ValueError(f"tracked-state prefix does not align with public features: {scope.query_id}")
        for slot in scope.tracked_slots:
            subject, component, allowed_mask = slot_binding(benchmark=benchmark, stream_id=scope.stream_id, slot=slot)
            examples.append(
                MemoryExample(
                    query_id=scope.query_id,
                    stream_id=scope.stream_id,
                    slot=slot,
                    features=stream_features[: scope.prefix_end_frame].contiguous(),
                    actions=actions[: scope.prefix_end_frame].contiguous(),
                    subject=subject,
                    component=component,
                    allowed_mask=allowed_mask,
                    target=0,
                )
            )
    return examples, scopes


@torch.inference_mode()
def run_memory_baseline(
    *, benchmark: PublicBenchmark, split: str, feature_root: str | Path, grounder_checkpoint: str | Path,
    checkpoint_root: str | Path, output_root: str | Path, device: str, batch_size: int, image_size: int,
    tracked_state_sidecar: str | Path | None = None,
) -> dict[str, object]:
    """Write state predictions without opening any evaluator-private file."""

    source_root, output = Path(checkpoint_root).resolve(), Path(output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output root is not empty: {output}")
    metadata = _read_json(source_root / "best_checkpoint_metadata.json")
    config_data = metadata.get("model_config")
    if not isinstance(config_data, dict):
        raise ValueError("checkpoint metadata lacks model_config")
    config = MemoryModelConfig(**config_data)
    model = MemoryStatePredictor(config)
    model.load_state_dict(load_file(str(source_root / "best.safetensors"), device="cpu"), strict=True)
    torch.backends.cudnn.enabled = False
    torch_device = torch.device(device)
    model.to(torch_device).eval()
    if tracked_state_sidecar is None:
        examples = build_public_examples(benchmark=benchmark, split=split, feature_root=feature_root,
                                         grounder_checkpoint=grounder_checkpoint, image_size=image_size)
        scopes = None
    else:
        examples, scopes = build_public_tracked_examples(
            benchmark=benchmark,
            split=split,
            feature_root=feature_root,
            grounder_checkpoint=grounder_checkpoint,
            image_size=image_size,
            tracked_state_sidecar=tracked_state_sidecar,
        )
    output.mkdir(parents=True, exist_ok=False)
    prediction_path = output / "state_predictions.jsonl"
    tracked_prediction_path = output / "tracked_state_predictions.jsonl"
    loader = DataLoader(MemoryDataset(examples), batch_size=batch_size, shuffle=False, collate_fn=collate_examples)
    completed = 0
    grouped: dict[str, dict[str, str | dict[str, str]]] = {}
    for batch in loader:
        logits = model(features=batch["features"].to(torch_device), actions=batch["actions"].to(torch_device),
                       lengths=batch["lengths"].to(torch_device), subject=batch["subject"].to(torch_device), component=batch["component"].to(torch_device))
        predictions = restrict_logits(logits, batch["component"].to(torch_device), batch["allowed_mask"].to(torch_device)).argmax(dim=-1).cpu().tolist()
        query_ids, slots = batch["query_ids"], batch["slots"]
        assert isinstance(query_ids, list) and isinstance(slots, list)
        for query_id, slot, prediction in zip(query_ids, slots, predictions, strict=True):
            query = benchmark.queries[str(query_id)]
            if scopes is None:
                _append_jsonl(prediction_path, {"query_id": query.query_id, "task": "state_query", "answer": decode_value(query=query, value_index=prediction, benchmark=benchmark)})
            else:
                entity_id, component = parse_slot_key(str(slot))
                state = grouped.setdefault(query.query_id, {})
                if slot in state:
                    raise ValueError("duplicate baseline prediction for tracked-state slot")
                state[str(slot)] = decode_slot_value(
                    benchmark=benchmark,
                    stream_id=query.stream_id,
                    entity_id=entity_id,
                    component=component,
                    value_index=int(prediction),
                )
            completed += 1
        _atomic_json(output / "progress.json", {"status": "running", "completed_slot_predictions": completed, "total_slot_predictions": len(examples)})
    if scopes is not None:
        for scope in sorted(scopes.for_split(split), key=lambda item: item.query_id):
            state = grouped.get(scope.query_id)
            if state is None or set(state) != set(scope.tracked_slots):
                raise ValueError(f"incomplete tracked-state baseline prediction: {scope.query_id}")
            _append_jsonl(
                tracked_prediction_path,
                {"query_id": scope.query_id, "task": TRACKED_STATE_TASK, "state": state},
            )
            _append_jsonl(
                prediction_path,
                {"query_id": scope.query_id, "task": "state_query", "answer": state[scope.target_slot]},
            )
    _atomic_json(output / "public_run_report.json", {
        "stage": "E2_visual_memory_form_public_inference", "status": "public_predictions_complete", "split": split,
        "method": config.method, "state_query_count": len(scopes.for_split(split)) if scopes is not None else completed,
        "method_inputs": ["frozen public visual transition features", "raw public actions", "public query and catalog"],
        "method_never_reads": ["decoded EventFacts", "Active State", "Runtime", "evaluator labels", "oracle state", "reveal RGB"],
        "grounder_checkpoint_sha256": sha256_file(grounder_checkpoint),
        "tracked_state_prediction_file": "tracked_state_predictions.jsonl" if scopes is not None else None,
        "tracked_slot_prediction_count": completed if scopes is not None else None,
    })
    _atomic_json(output / "progress.json", {
        "status": "public_predictions_complete",
        "state_query_count": len(scopes.for_split(split)) if scopes is not None else completed,
        "tracked_slot_prediction_count": completed if scopes is not None else None,
    })
    return {
        "status": "passed",
        "output_root": str(output),
        "state_query_count": len(scopes.for_split(split)) if scopes is not None else completed,
        "tracked_slot_prediction_count": completed if scopes is not None else None,
        "method": config.method,
    }
