"""Public-only inference for frozen E2 next-frame renderers.

This module intentionally does not import the train/validation data assembly
or any evaluator module.  It can therefore be run on test without opening a
target image, source mask, counterfactual rendering, state answer, or plan.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset

from e0.public_protocol import PublicBenchmark
from e0_5.learned_grounder import SUBJECT_INDEX, VALUE_INDEX, visual_subject_key
from e2_memory.features import load_feature_cache

from .p1_conditions import P1_SCHEMA_VERSION
from .public_odometry import PublicVisualOdometry, translate_public_binding
from .renderer import (
    COMPONENT_ORDER, DEFAULT_INTERFACE_VARIANT, RESIDUAL_METHODS,
    RendererConfig, build_renderer, interface_uses_alignment, interface_uses_binding, interface_uses_state,
)


INFERENCE_VERSION = "statereturn-visual-next-frame-inference/v2"


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    Image.fromarray(image).save(temporary)
    os.replace(temporary, path)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL object required: {path}:{number}")
        rows.append(value)
    return rows


def _rgb(path: Path, size: int) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB").resize((size, size), Image.Resampling.NEAREST)).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(127.5).sub_(1.0)


def _catalog(benchmark: PublicBenchmark, stream_id: str, entity_id: str) -> dict[str, object]:
    for item in benchmark.streams[stream_id].entity_catalog:
        if item.get("entity_id") == entity_id:
            return item
    raise ValueError(f"unknown public entity: {stream_id}/{entity_id}")


def _value_index(*, benchmark: PublicBenchmark, stream_id: str, component: str, value: object) -> int:
    if component == "place":
        if not isinstance(value, dict) or value.get("kind") != "in_zone" or not isinstance(value.get("target"), str):
            raise ValueError("public WCM place condition is malformed")
        value = _catalog(benchmark, stream_id, str(value["target"])).get("pattern")
    if value not in VALUE_INDEX:
        raise ValueError(f"unsupported public WCM state value: {value!r}")
    return VALUE_INDEX[str(value)]


class _Example:
    def __init__(self, *, query_id: str, current_path: Path, features: torch.Tensor | None, actions: torch.Tensor | None,
                 binding_path: Path | None, subject: int | None,
                 component: int | None, value: int | None,
                 binding_projection: tuple[int, int, bool] | None,
                 requires_binding_projection: bool) -> None:
        self.query_id, self.current_path = query_id, current_path
        self.features, self.actions = features, actions
        self.binding_path = binding_path
        self.subject, self.component, self.value = subject, component, value
        self.binding_projection = binding_projection
        self.requires_binding_projection = requires_binding_projection


class _Dataset(Dataset[_Example]):
    def __init__(self, rows: list[_Example]) -> None: self.rows = rows
    def __len__(self) -> int: return len(self.rows)
    def __getitem__(self, index: int) -> _Example: return self.rows[index]


def _collate(rows: list[_Example], image_size: int) -> dict[str, torch.Tensor | list[str]]:
    current = torch.stack([_rgb(row.current_path, image_size) for row in rows])
    binding = torch.stack([_rgb(row.binding_path, image_size) if row.binding_path else torch.zeros_like(current[0]) for row in rows])
    result: dict[str, torch.Tensor | list[str]] = {"query_ids": [row.query_id for row in rows], "current": current, "binding": binding}
    if any(row.features is not None for row in rows):
        assert all(row.features is not None and row.actions is not None for row in rows)
        lengths = torch.tensor([row.features.shape[0] for row in rows], dtype=torch.long)
        features = torch.zeros(len(rows), int(lengths.max()), rows[0].features.shape[1]); actions = torch.zeros(len(rows), int(lengths.max()), dtype=torch.long)
        for index, row in enumerate(rows):
            assert row.features is not None and row.actions is not None
            features[index, :row.features.shape[0]] = row.features; actions[index, :row.actions.shape[0]] = row.actions
        result.update({"features":features, "actions":actions, "lengths":lengths})
    if any(row.subject is not None for row in rows):
        assert all(row.subject is not None and row.component is not None for row in rows)
        result.update({"subject":torch.tensor([int(row.subject) for row in rows]), "component":torch.tensor([int(row.component) for row in rows])})
    if any(row.value is not None for row in rows):
        assert all(row.value is not None for row in rows)
        result.update({"value":torch.tensor([int(row.value) for row in rows]), "binding_present":torch.tensor([row.binding_path is not None for row in rows])})
    if any(row.requires_binding_projection for row in rows):
        assert all(row.requires_binding_projection for row in rows)
        projected: list[torch.Tensor] = []
        availability: list[bool] = []
        if any(row.binding_projection is not None for row in rows):
            assert all(row.binding_projection is not None for row in rows)
            for index, row in enumerate(rows):
                assert row.binding_projection is not None
                delta_y, delta_x, available = row.binding_projection
                projected.append(translate_public_binding(binding[index], delta_y=delta_y, delta_x=delta_x))
                availability.append(available)
        else:
            projected = [torch.zeros_like(current[0]) for _ in rows]
            availability = [False for _ in rows]
        result.update({"binding_projection":torch.stack(projected), "projection_present":torch.tensor(availability)})
    return result


def _public_examples(*, benchmark: PublicBenchmark, dataset_root: Path, p0_root: Path, split: str, method: str,
                     feature_root: Path | None, grounder_checkpoint: Path | None, feature_image_size: int,
                     wcm_condition_root: Path | None,
                     interface_variant: str = DEFAULT_INTERFACE_VARIANT,
                     query_slot_conditioning: bool = False) -> list[_Example]:
    p0_rows = {str(row["query_id"]): row for row in _read_jsonl(p0_root / "public" / f"{split}.jsonl")}
    expected = {query.query_id for query in benchmark.queries.values() if query.task == "next_frame" and query.split == split}
    if set(p0_rows) != expected:
        raise ValueError("P0 public test index has incomplete coverage")
    if method != "wcm_odometry_residual" and interface_variant != DEFAULT_INTERFACE_VARIANT:
        raise ValueError("interface ablations require wcm_odometry_residual")
    if query_slot_conditioning and method == "wcm_odometry_residual":
        raise ValueError("query-slot conditioning is defined only for history renderers")
    uses_wcm = method == "wcm_odometry_residual"
    uses_history = True
    is_interface_ablation = uses_wcm
    state_enabled = interface_uses_state(interface_variant) if is_interface_ablation else True
    binding_enabled = interface_uses_binding(interface_variant) if is_interface_ablation else True
    alignment_enabled = interface_uses_alignment(interface_variant) if is_interface_ablation else False
    odometry = PublicVisualOdometry(benchmark=benchmark, dataset_root=dataset_root) if method == "wcm_odometry_residual" and alignment_enabled else None
    if uses_wcm and state_enabled:
        if wcm_condition_root is None: raise ValueError("WCM inference requires P1 condition root")
        conditions = {str(row["query_id"]):row for row in _read_jsonl(wcm_condition_root / "conditions" / f"{split}.jsonl")}
        if set(conditions) != expected: raise ValueError("P1 WCM conditions have incomplete coverage")
    else:
        conditions = {}
    if uses_history:
        if feature_root is None or grounder_checkpoint is None: raise ValueError("history-aware inference requires public feature cache")
        features: dict[str, tuple[torch.Tensor,torch.Tensor]] | None = load_feature_cache(root=feature_root, benchmark=benchmark, split=split, checkpoint_path=grounder_checkpoint, image_size=feature_image_size)
    else:
        features = None
    rows: list[_Example] = []
    for query_id in sorted(expected):
        query, p0 = benchmark.queries[query_id], p0_rows[query_id]
        current = p0.get("current_rgb_path")
        if not isinstance(current, str) or p0.get("next_action") != "forward": raise ValueError("malformed public P0 query")
        feature, actions = None, None
        if uses_history:
            assert features is not None; feature, actions = features[query.stream_id]
            feature, actions = feature[:query.prefix_end_frame].contiguous(), actions[:query.prefix_end_frame].contiguous()
        item = _catalog(benchmark, query.stream_id, query.entity_id)
        subject = SUBJECT_INDEX[visual_subject_key(item)]
        component = COMPONENT_ORDER.index(query.component)
        if not uses_wcm:
            rows.append(_Example(query_id=query_id,current_path=dataset_root/"public"/current,features=feature,actions=actions,binding_path=None,
                                 subject=subject if query_slot_conditioning else None,
                                 component=component if query_slot_conditioning else None,
                                 value=None,binding_projection=None,requires_binding_projection=False)); continue
        if is_interface_ablation and not state_enabled:
            rows.append(_Example(query_id=query_id,current_path=dataset_root/"public"/current,features=feature,actions=actions,binding_path=None,subject=0,component=0,value=0,binding_projection=None,requires_binding_projection=True)); continue
        condition = conditions[query_id]; compiled = condition.get("compiled_query_state"); binding = condition.get("visual_binding") if binding_enabled else None
        if condition.get("schema_version") != P1_SCHEMA_VERSION or condition.get("query_slot") != f"{query.entity_id}|{query.component}" or not isinstance(compiled,dict): raise ValueError("P1 test condition mismatch")
        if compiled.get("entity_id") != query.entity_id or compiled.get("component") != query.component: raise ValueError("P1 query-state identity mismatch")
        binding_path = None
        projection_after_frame_index: int | None = None
        binding_projection: tuple[int, int, bool] | None = None
        if isinstance(binding,dict):
            rel=binding.get("after_rgb_path")
            if not isinstance(rel,str): raise ValueError("P1 binding path is malformed")
            binding_path=dataset_root/"public"/rel
            if alignment_enabled:
                after = binding.get("after_frame_index")
                if isinstance(after, bool) or not isinstance(after, int) or after < 0 or after >= query.prefix_end_frame:
                    raise ValueError("P1 binding after-frame is not a valid public transition-trace anchor")
                projection_after_frame_index = after
        elif alignment_enabled:
            raise ValueError("odometry residual renderer requires a public visual binding")
        if method == "wcm_odometry_residual" and alignment_enabled:
            assert odometry is not None and projection_after_frame_index is not None and query.next_action is not None
            projection = odometry.binding_to_next_view(
                stream_id=query.stream_id, binding_after_frame_index=projection_after_frame_index,
                prefix_end_frame=query.prefix_end_frame, next_action=query.next_action,
            )
            binding_projection = (projection.delta_y, projection.delta_x, projection.available)
        rows.append(_Example(query_id=query_id,current_path=dataset_root/"public"/current,features=feature,actions=actions,binding_path=binding_path,
                             subject=subject,component=component,
                             value=_value_index(benchmark=benchmark,stream_id=query.stream_id,component=query.component,value=compiled.get("value")),
                             binding_projection=binding_projection,requires_binding_projection=method == "wcm_odometry_residual"))
    return rows


@torch.inference_mode()
def run_renderer(*, benchmark: PublicBenchmark, dataset_root: str | Path, p0_root: str | Path, split: str,
                 checkpoint_root: str | Path, output_root: str | Path, device: str, batch_size: int,
                 feature_root: str | Path | None = None, grounder_checkpoint: str | Path | None = None,
                 feature_image_size: int = 160, wcm_condition_root: str | Path | None = None) -> dict[str, object]:
    if split not in {"validation", "test"}:
        raise ValueError("formal inference is defined for validation or test only")
    source, output, dataset, p0 = Path(checkpoint_root).resolve(), Path(output_root).resolve(), Path(dataset_root).resolve(), Path(p0_root).resolve()
    if output.exists() and any(output.iterdir()): raise FileExistsError(f"inference output exists: {output}")
    metadata=json.loads((source/"best_checkpoint_metadata.json").read_text()); config_data=metadata.get("renderer_config")
    if not isinstance(config_data,dict): raise ValueError("checkpoint lacks renderer config")
    config=RendererConfig(**config_data)
    base_data=metadata.get("base_renderer_config")
    if config.method in RESIDUAL_METHODS:
        if not isinstance(base_data, dict): raise ValueError("WCM residual checkpoint lacks frozen base config")
        base_config=RendererConfig(**base_data)
    else:
        if base_data is not None: raise ValueError("non-residual checkpoint must not declare a frozen base config")
        base_config=None
    image_size=json.loads((source/"training_protocol.json").read_text()).get("image_size")
    if not isinstance(image_size,int) or image_size<=0: raise ValueError("checkpoint protocol lacks image size")
    rows=_public_examples(benchmark=benchmark,dataset_root=dataset,p0_root=p0,split=split,method=config.method,feature_root=Path(feature_root).resolve() if feature_root else None,grounder_checkpoint=Path(grounder_checkpoint).resolve() if grounder_checkpoint else None,feature_image_size=feature_image_size,wcm_condition_root=Path(wcm_condition_root).resolve() if wcm_condition_root else None,interface_variant=config.interface_variant,query_slot_conditioning=config.query_slot_conditioning)
    model=build_renderer(config=config,base_config=base_config); model.load_state_dict(load_file(str(source/"best.safetensors"),device="cpu"),strict=True); torch.backends.cudnn.enabled=False; torch_device=torch.device(device); model.to(torch_device).eval()
    output.mkdir(parents=True,exist_ok=False); loader=DataLoader(_Dataset(rows),batch_size=batch_size,shuffle=False,num_workers=2,pin_memory=True,collate_fn=lambda batch:_collate(batch,image_size)); prediction_rows=[]
    for batch in loader:
        current=batch["current"].to(torch_device); binding=batch["binding"].to(torch_device); kwargs={key:batch[key].to(torch_device) for key in ("features","actions","lengths","subject","component","value","binding_present","binding_projection","projection_present") if key in batch}
        prediction=model(current=current,binding=binding,**kwargs).cpu().add(1).mul(127.5).round().clamp(0,255).byte().permute(0,2,3,1).numpy()
        for query_id,image in zip(batch["query_ids"],prediction,strict=True):
            relative=f"png/{query_id}.png"; _atomic_png(output/relative,image); prediction_rows.append({"query_id":query_id,"task":"next_frame","png_path":relative})
        _atomic_json(output/"progress.json",{"status":"running","completed_predictions":len(prediction_rows),"total_predictions":len(rows)})
    payload="".join(json.dumps(row,sort_keys=True)+"\n" for row in prediction_rows); (output/"predictions.jsonl").write_text(payload,encoding="utf-8")
    report={"version":INFERENCE_VERSION,"status":"public_predictions_complete","split":split,"method":config.method,"prediction_count":len(prediction_rows),"renderer_config":asdict(config),"method_never_reads":["test target RGB","counterfactual RGB","source masks","state labels","Oracle state","simulator metadata"]}
    if config.method != "wcm_odometry_residual":
        report["public_query_slot"]={
            "enabled":config.query_slot_conditioning,
            "fields":["entity_id","component"] if config.query_slot_conditioning else [],
            "does_not_include":["active-state value","visual binding","binding projection"],
        }
    if config.method == "wcm_odometry_residual":
        report["interface_ablation"]={
            "variant":config.interface_variant,
            "active_state":interface_uses_state(config.interface_variant),
            "visual_binding":interface_uses_binding(config.interface_variant),
            "binding_alignment":interface_uses_alignment(config.interface_variant),
            "history_only_condition_access":"P1 state/binding condition files are not opened" if config.interface_variant == "history_only" else "P1 condition fields are masked according to the variant",
        }
    _atomic_json(output/"public_run_report.json",report)
    _atomic_json(output/"progress.json",{"status":"public_predictions_complete","completed_predictions":len(prediction_rows),"total_predictions":len(rows)})
    return {"status":"passed","method":config.method,"prediction_count":len(prediction_rows),"output_root":str(output)}
