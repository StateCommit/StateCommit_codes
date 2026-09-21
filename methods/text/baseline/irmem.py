"""Public-only online implicit recurrent-memory baseline."""

from __future__ import annotations

import gc
import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from e1core.answer_codec import UNTRACKED
from e1core.protocol import BASE_COMPONENTS, NaturalEvent, PublicEvent, QueryExample, REPRESENTATIONS, StructuredEvent, canonical_json, load_examples, read_jsonl, sha256_file
from e1core.transition_spec import spec_provenance

from .base import Generation, MemoryMethod


IRMEM_VERSION = "statereturn-text-irmem/v1"
SAFE_FORMAT_VERSION = "statereturn-safetensors-json/v1"
PLACE_KINDS = ("on", "inside", "held_by")
TRAINING_SPLITS = ("train", "validation")


PLACE_KIND_LOSS_WEIGHT = 1.0
POINTER_LOSS_WEIGHT = 0.5
LOSS_VERSION = "component-balanced-separate-weights-v2"


TRAIN_BATCH_SIZE = 16
GRAD_CLIP_NORM = 5.0
LABEL_SMOOTHING = 0.05
KEY_DIM = 128
LR_SCHEDULER_PATIENCE = 5
LR_SCHEDULER_FACTOR = 0.5


def _torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError("IRMem requires torch") from exc
    return torch, nn


def _safe_tensor_io():
    """Import safe tensor I/O only when IRMem is actually used."""

    try:
        from safetensors.torch import load_file, save_file
    except ImportError as exc:
        raise RuntimeError("IRMem requires safetensors; unsafe pickle checkpoints are not supported") from exc
    return load_file, save_file


def _atomic_json(path: Path, value: Mapping[str, Any] | Sequence[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _device_memory(device: str) -> dict[str, int | str | None]:
    try:
        torch, _nn = _torch()
        if device.startswith("cuda") and torch.cuda.is_available():
            return {
                "device": device,
                "allocated_bytes": int(torch.cuda.memory_allocated(device)),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            }
    except (RuntimeError, ValueError):
        pass
    return {"device": device, "allocated_bytes": None, "peak_allocated_bytes": None}


def _reset_device_peak(device: str) -> None:
    try:
        torch, _nn = _torch()
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
    except (RuntimeError, ValueError):
        return


class FrozenQwenEncoder:
    """Frozen mean-pooled Qwen encoder with explicit online-compute accounting."""

    def __init__(self, *, model_path: str | Path, device: str) -> None:
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("IRMem requires transformers") from exc
        torch, _nn = _torch()
        self.torch, self.device = torch, device
        self.model_path = str(Path(model_path).resolve())
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModel.from_pretrained(
            self.model_path, torch_dtype=dtype, trust_remote_code=True
        ).to(device).eval()
        self.model.requires_grad_(False)
        self.dimension = int(getattr(self.model.config, "hidden_size"))
        self._usage = {"encoder_token_count": 0, "encoder_forward_count": 0, "encoder_seconds": 0.0}

    def encode(self, texts: Sequence[str], *, batch_size: int = 16):
        vectors = []
        with self.torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                encoded = self.tokenizer(
                    list(texts[start : start + batch_size]),
                    return_tensors="pt", padding=True, truncation=True, max_length=512,
                ).to(self.device)
                started = time.monotonic()
                hidden = self.model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)

                vectors.append(pooled.float().cpu())
                self._usage["encoder_seconds"] += time.monotonic() - started
                self._usage["encoder_token_count"] += int(encoded["attention_mask"].sum().item())
                self._usage["encoder_forward_count"] += 1
        return self.torch.cat(vectors, dim=0) if vectors else self.torch.empty((0, self.dimension))

    def usage(self) -> dict[str, int | float]:
        return {
            "encoder_token_count": int(self._usage["encoder_token_count"]),
            "encoder_forward_count": int(self._usage["encoder_forward_count"]),
            "encoder_seconds": round(float(self._usage["encoder_seconds"]), 6),
        }

    def close(self) -> None:
        """Release the 7B encoder before GRU-only training begins."""

        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        gc.collect()
        if self.device.startswith("cuda") and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def _network(input_dim: int, hidden_size: int, *, key_dim: int = KEY_DIM):

    torch, nn = _torch()
    adapter_dim = max(32, hidden_size // 4)

    class Network(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.hidden_size = hidden_size




            self.event_norm = nn.LayerNorm(input_dim)
            self.query_norm = nn.LayerNorm(input_dim)
            self.entity_norm = nn.LayerNorm(input_dim)
            self.cell = nn.GRUCell(input_dim, hidden_size)

            self.hidden_norm = nn.LayerNorm(hidden_size)
            self.query_proj = nn.Sequential(nn.Linear(input_dim, hidden_size), nn.GELU())
            self.context = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.GELU())


            self.openness_adapter = nn.Sequential(nn.Linear(hidden_size, adapter_dim), nn.GELU())
            self.toggle_adapter = nn.Sequential(nn.Linear(hidden_size, adapter_dim), nn.GELU())
            self.place_kind_adapter = nn.Sequential(nn.Linear(hidden_size, adapter_dim), nn.GELU())
            self.pointer_adapter = nn.Sequential(nn.Linear(hidden_size, adapter_dim), nn.GELU())


            self.pointer_query_proj = nn.Linear(adapter_dim, key_dim)
            self.pointer_key_proj = nn.Linear(input_dim, key_dim)


            self.openness_head = nn.Linear(adapter_dim, 2)
            self.toggle_head = nn.Linear(adapter_dim, 2)
            self.place_kind_head = nn.Linear(adapter_dim, len(PLACE_KINDS))

            self._init_weights()

        def _init_weights(self) -> None:
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)



            for name, param in self.cell.named_parameters():
                if "bias_ih" in name or "bias_hh" in name:
                    gate_dim = param.size(0) // 3

                    param.data[gate_dim * 2 :].uniform_(-1.0, 0.0)

                    param.data[gate_dim : gate_dim * 2].uniform_(0.5, 1.0)


        def initial(self, *, device: str):
            return torch.zeros((1, hidden_size), device=device)

        def update(self, event, hidden):
            new_hidden = self.cell(self.event_norm(event).unsqueeze(0), hidden)
            return self.hidden_norm(new_hidden)

        def context_for(self, hidden, query):
            return self.context(torch.cat((hidden, self.query_proj(self.query_norm(query).unsqueeze(0))), dim=-1))

        def logits(self, hidden, query, candidates):
            shared = self.context_for(hidden, query)

            openness_feat = self.openness_adapter(shared)
            toggle_feat = self.toggle_adapter(shared)
            place_kind_feat = self.place_kind_adapter(shared)
            pointer_feat = self.pointer_adapter(shared)


            pointer_query = self.pointer_query_proj(pointer_feat)
            candidate_keys = self.pointer_key_proj(
                self.entity_norm(candidates.to(shared.device))
            )
            pointer_logits = (
                candidate_keys @ pointer_query.squeeze(0)
            ) / (key_dim ** 0.5)

            return {
                "openness": self.openness_head(openness_feat),
                "toggle": self.toggle_head(toggle_feat),
                "place_kind": self.place_kind_head(place_kind_feat),
                "pointer": pointer_logits.unsqueeze(0),
            }

    return Network()


def _event_text(event: PublicEvent) -> str:
    """Losslessly present the *public* event semantics to the frozen encoder.

    ``event_id`` is an opaque trace identifier.  Encoding it alongside the short
    action sentence made its random hash tokens dominate mean pooling and gave
    the recurrent baseline little usable action signal.  It is intentionally
    omitted here: it remains a cache key, never a semantic input.
    """

    if isinstance(event, StructuredEvent):
        arguments = "; ".join(f"{key}={event.arguments[key]}" for key in sorted(event.arguments))
        lines = [
            "Representation: structured event.",
            f"Action: {event.action_type}.",
            f"Arguments: {arguments}.",
            f"Outcome: {event.outcome}.",
        ]
        if event.place_effect is not None:
            effect = event.place_effect
            lines.append(
                "Public placement observation: "
                f"entity={effect.entity_id}; relation={effect.kind}; target={effect.target}."
            )
        return "\n".join(lines)
    if isinstance(event, NaturalEvent):
        return "\n".join((
            "Representation: natural-language event.",
            f"Description: {event.description}",
            f"Outcome observation: {event.outcome_text}",
        ))
    raise TypeError(f"unsupported public event: {type(event)!r}")


def _query_text(entity_id: str, component: str) -> str:
    return f"World entity: {entity_id}.\nRequested state component: {component}."


def _entity_text(entity_id: str) -> str:
    """Encode the public entity ID with its type prefix for semantic grounding.

    Entity IDs follow the pattern ``<type>_<number>`` (e.g. ``box_1``,
    ``table_3``, ``agent_1``).  The type prefix is public — it appears in
    event descriptions and arguments — and does not reveal any state value.
    """
    parts = entity_id.rsplit("_", 1)
    entity_type = parts[0] if len(parts) == 2 and parts[1].isdigit() else "entity"
    return f"Entity {entity_id} of kind {entity_type}."


def _entities(events: Iterable[PublicEvent]) -> tuple[str, ...]:
    ids = {"agent_1"}
    for event in events:
        ids.update(event.entity_ids())
    return tuple(sorted(ids))


def _labels(dataset_root: str | Path, *, split: str) -> dict[str, Any]:
    if split not in TRAINING_SPLITS:
        raise RuntimeError("IRMem may load labels only for train or validation")
    result: dict[str, Any] = {}
    for row in read_jsonl(Path(dataset_root) / "labels" / f"{split}.jsonl"):
        query_id = row.get("query_id")
        if not isinstance(query_id, str) or "answer" not in row or query_id in result:
            raise RuntimeError("IRMem needs answer-only public train/validation labels")
        result[query_id] = row["answer"]
    return result


def _model_files(model_path: str | Path) -> list[Path]:
    """Return every file that can affect Qwen weights or tokenization."""

    root = Path(model_path)
    if not root.is_dir():
        raise RuntimeError(f"IRMem model path does not exist: {root}")
    files = sorted(
        path for path in root.iterdir()
        if path.is_file()
        and (
            path.suffix in {".safetensors", ".bin", ".model"}
            or path.name in {"config.json", "configuration.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"}
        )
    )
    if not any(path.suffix in {".safetensors", ".bin"} for path in files):
        raise RuntimeError("IRMem model fingerprint found no weight files")
    return files


def _encoder_fingerprint(model_path: str | Path) -> str:
    """Content hash of all local model weights and tokenizer-defining files."""

    root = Path(model_path)
    files = [
        {"name": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in _model_files(root)
    ]
    return hashlib.sha256(json.dumps(files, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _model_dimension(model_path: str | Path) -> int:
    try:
        value = json.loads((Path(model_path) / "config.json").read_text(encoding="utf-8"))
        dimension = value.get("hidden_size")
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("IRMem cannot read model hidden_size from config.json") from exc
    if not isinstance(dimension, int) or dimension <= 0:
        raise RuntimeError("IRMem model config has invalid hidden_size")
    return dimension


def _public_input_fingerprints(dataset_root: Path, *, splits: Sequence[str], representation: str = "structured") -> dict[str, str]:
    if set(splits) - set(TRAINING_SPLITS):
        raise RuntimeError("IRMem training cache may not include test inputs")
    if representation not in REPRESENTATIONS:
        raise RuntimeError(f"IRMem has unsupported representation: {representation}")
    return {split: sha256_file(dataset_root / "data" / representation / f"{split}.jsonl") for split in splits}


@dataclass(frozen=True, slots=True)
class EmbeddingCache:
    key: str
    input_dim: int
    events: Mapping[str, Any]
    queries: Mapping[str, Any]
    entities: Mapping[str, Any]
    metadata: Mapping[str, Any]


def _cache_key(*, input_fingerprints: Mapping[str, str], encoder_fingerprint: str, representation: str) -> str:
    payload = {
        "format_version": SAFE_FORMAT_VERSION,
        "irmem_version": IRMEM_VERSION,
        "splits": dict(sorted(input_fingerprints.items())),
        "encoder_fingerprint": encoder_fingerprint,
        "representation": representation,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _cache_paths(cache_root: Path, key: str) -> tuple[Path, Path]:
    stem = cache_root / f"public-train-validation-{key}"
    return stem.with_suffix(".safetensors"), stem.with_suffix(".metadata.json")


def _cache_maps_from_tensors(*, tensors: Mapping[str, Any], metadata: Mapping[str, Any], key: str) -> EmbeddingCache:
    torch, _nn = _torch()
    required = {"event_vectors", "query_vectors", "entity_vectors"}
    if set(tensors) != required:
        raise RuntimeError("IRMem embedding cache tensor keys are malformed")
    indexes = metadata.get("indexes")
    if not isinstance(indexes, Mapping):
        raise RuntimeError("IRMem embedding cache lacks indexes")
    dimension = metadata.get("input_dim")
    if not isinstance(dimension, int) or dimension <= 0:
        raise RuntimeError("IRMem embedding cache has invalid input_dim")

    def build(kind: str, tensor_name: str) -> dict[str, Any]:
        keys = indexes.get(kind)
        tensor = tensors[tensor_name]
        if not isinstance(keys, list) or not all(isinstance(item, str) for item in keys):
            raise RuntimeError(f"IRMem embedding cache has invalid {kind} index")
        if tensor.ndim != 2 or tensor.shape != (len(keys), dimension):
            raise RuntimeError(f"IRMem embedding cache has invalid {kind} tensor shape")
        return {item: tensor[index].detach().cpu() for index, item in enumerate(keys)}

    return EmbeddingCache(
        key=key,
        input_dim=dimension,
        events=build("events", "event_vectors"),
        queries=build("queries", "query_vectors"),
        entities=build("entities", "entity_vectors"),
        metadata=metadata,
    )


def _load_embedding_cache(
    *, cache_root: Path, key: str, expected_metadata: Mapping[str, Any]
) -> EmbeddingCache | None:
    tensor_path, metadata_path = _cache_paths(cache_root, key)
    if not tensor_path.is_file() and not metadata_path.is_file():
        return None
    if not tensor_path.is_file() or not metadata_path.is_file():
        raise RuntimeError("IRMem embedding cache is incomplete; remove the two cache files and rebuild")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("IRMem embedding cache metadata is unreadable") from exc
    required_keys = set(expected_metadata) | {"indexes"}
    if set(metadata) != required_keys or any(metadata.get(key) != value for key, value in expected_metadata.items()):
        raise RuntimeError("IRMem embedding cache provenance mismatch")
    load_file, _save_file = _safe_tensor_io()
    return _cache_maps_from_tensors(tensors=load_file(str(tensor_path), device="cpu"), metadata=metadata, key=key)


def _build_embedding_cache(
    *, dataset_root: Path, cache_root: Path, key: str, metadata: Mapping[str, Any], representation: str, encoder: FrozenQwenEncoder
) -> EmbeddingCache:
    """Encode train/validation public inputs only, then persist a safe cache."""

    event_texts: dict[str, str] = {}
    query_texts: dict[str, str] = {}
    entity_texts: dict[str, str] = {"agent_1": _entity_text("agent_1")}
    for split in TRAINING_SPLITS:
        for example in load_examples(dataset_root, split=split, representation=representation):
            query_key = _query_text(example.target_entity_id, example.target_component)
            query_texts[query_key] = query_key
            for event in example.history:
                text = _event_text(event)
                prior = event_texts.setdefault(event.event_id, text)
                if prior != text:
                    raise RuntimeError(f"public event_id maps to non-identical text: {event.event_id}")
                for entity in _entities((event,)):
                    entity_texts[entity] = _entity_text(entity)

    def encode_index(items: Mapping[str, str]) -> tuple[list[str], Any]:
        keys = sorted(items)
        return keys, encoder.encode([items[key] for key in keys])

    event_keys, event_vectors = encode_index(event_texts)
    query_keys, query_vectors = encode_index(query_texts)
    entity_keys, entity_vectors = encode_index(entity_texts)
    metadata = dict(metadata)
    metadata["indexes"] = {"events": event_keys, "queries": query_keys, "entities": entity_keys}
    cache_root.mkdir(parents=True, exist_ok=True)
    tensor_path, metadata_path = _cache_paths(cache_root, key)
    _load_file, save_file = _safe_tensor_io()
    save_file(
        {
            "event_vectors": event_vectors.contiguous(),
            "query_vectors": query_vectors.contiguous(),
            "entity_vectors": entity_vectors.contiguous(),
        },
        str(tensor_path),
        metadata={"format_version": SAFE_FORMAT_VERSION, "kind": "public_train_validation_embeddings"},
    )
    _atomic_json(metadata_path, metadata)
    return _cache_maps_from_tensors(
        tensors={"event_vectors": event_vectors, "query_vectors": query_vectors, "entity_vectors": entity_vectors},
        metadata=metadata,
        key=key,
    )


def _embedding_cache(
    *, dataset_root: Path, model_path: str | Path, cache_root: Path, encoder_fingerprint: str,
    representation: str, encoder: FrozenQwenEncoder | None,
) -> EmbeddingCache:
    """Load a safe train/validation-only cache, building it only on cache miss."""

    input_fingerprints = _public_input_fingerprints(dataset_root, splits=TRAINING_SPLITS, representation=representation)
    key = _cache_key(input_fingerprints=input_fingerprints, encoder_fingerprint=encoder_fingerprint, representation=representation)
    expected_metadata: dict[str, Any] = {
        "format_version": SAFE_FORMAT_VERSION,
        "irmem_version": IRMEM_VERSION,
        "cache_key": key,
        "encoder_fingerprint": encoder_fingerprint,
        "input_dim": _model_dimension(model_path),
        "public_input_splits": list(TRAINING_SPLITS),
        "public_input_sha256": input_fingerprints,
        "test_input_access_during_training": False,
        "representation": representation,
    }
    cached = _load_embedding_cache(cache_root=cache_root, key=key, expected_metadata=expected_metadata)
    if cached is not None:
        return cached
    if encoder is None:
        raise RuntimeError("IRMem train/validation embedding cache miss requires an encoder")
    if encoder.dimension != expected_metadata["input_dim"]:
        raise RuntimeError("IRMem encoder hidden dimension differs from model config")
    return _build_embedding_cache(
        dataset_root=dataset_root, cache_root=cache_root, key=key, metadata=expected_metadata, representation=representation, encoder=encoder
    )


def _sample_loss(
    network, item: tuple[QueryExample, Any], embeddings: EmbeddingCache, *,
    device: str, label_smoothing: float = LABEL_SMOOTHING,
):

    torch, _nn = _torch()
    example, answer = item
    hidden = network.initial(device=device)
    for event in example.history:
        hidden = network.update(embeddings.events[event.event_id].to(device), hidden)
    entities = _entities(example.history)
    candidates = torch.stack([embeddings.entities[entity] for entity in entities])
    logits = network.logits(hidden, embeddings.queries[_query_text(example.target_entity_id, example.target_component)].to(device), candidates)
    if example.target_component == "openness":
        return torch.nn.functional.cross_entropy(
            logits["openness"], torch.tensor([("open", "closed").index(answer)], device=device),
            label_smoothing=label_smoothing,
        )
    if example.target_component == "toggle_state":
        return torch.nn.functional.cross_entropy(
            logits["toggle"], torch.tensor([("on", "off").index(answer)], device=device),
            label_smoothing=label_smoothing,
        )
    if not isinstance(answer, Mapping) or answer.get("kind") not in PLACE_KINDS or answer.get("target") not in entities:
        raise RuntimeError("public place label is malformed for IRMem")
    kind_loss = torch.nn.functional.cross_entropy(
        logits["place_kind"], torch.tensor([PLACE_KINDS.index(answer["kind"])], device=device),
        label_smoothing=label_smoothing,
    )
    target_loss = torch.nn.functional.cross_entropy(
        logits["pointer"], torch.tensor([entities.index(answer["target"])], device=device),
        label_smoothing=0.0,
    )
    return PLACE_KIND_LOSS_WEIGHT * kind_loss + POINTER_LOSS_WEIGHT * target_loss


def _train_epoch(
    network, optimizer, samples: Sequence[tuple[QueryExample, Any]], embeddings: EmbeddingCache, *,
    device: str, label_smoothing: float = LABEL_SMOOTHING,
) -> list[float]:
    """Train on small, mixed batches to avoid one-component update oscillation."""

    torch, _nn = _torch()
    losses: list[float] = []
    for start in range(0, len(samples), TRAIN_BATCH_SIZE):
        batch = samples[start : start + TRAIN_BATCH_SIZE]
        optimizer.zero_grad(set_to_none=True)
        loss = torch.stack([
            _sample_loss(network, item, embeddings, device=device, label_smoothing=label_smoothing)
            for item in batch
        ]).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return losses


def _predict(network, example: QueryExample, embeddings: EmbeddingCache, *, device: str) -> Any:
    torch, _nn = _torch()
    hidden = network.initial(device=device)
    for event in example.history:
        hidden = network.update(embeddings.events[event.event_id].to(device), hidden)
    entities = _entities(example.history)
    candidates = torch.stack([embeddings.entities[entity] for entity in entities])
    logits = network.logits(hidden, embeddings.queries[_query_text(example.target_entity_id, example.target_component)].to(device), candidates)
    if example.target_component == "openness":
        return ("open", "closed")[int(logits["openness"].argmax(dim=-1).item())]
    if example.target_component == "toggle_state":
        return ("on", "off")[int(logits["toggle"].argmax(dim=-1).item())]
    return {
        "kind": PLACE_KINDS[int(logits["place_kind"].argmax(dim=-1).item())],
        "target": entities[int(logits["pointer"].argmax(dim=-1).item())],
    }


def _prediction_diagnostics(network, samples: Sequence[tuple[QueryExample, Any]], embeddings: EmbeddingCache, *, device: str) -> dict[str, Any]:
    """Measure whether every IRMem head uses history instead of a fixed prior.

    All labels passed here are public train/validation labels.  This helper is
    deliberately unavailable for test because it is used only while choosing
    an IRMem checkpoint.
    """

    torch, _nn = _torch()
    by_component: dict[str, dict[str, Any]] = {
        component: {"count": 0, "correct": 0, "prediction_distribution": {}, "confusion": {}}
        for component in BASE_COMPONENTS
    }
    with torch.inference_mode():
        for example, answer in samples:
            prediction = _predict(network, example, embeddings, device=device)
            component = example.target_component
            expected_key, predicted_key = canonical_json(answer), canonical_json(prediction)
            item = by_component[component]
            item["count"] += 1
            item["correct"] += int(expected_key == predicted_key)
            item["prediction_distribution"][predicted_key] = item["prediction_distribution"].get(predicted_key, 0) + 1
            item["confusion"].setdefault(expected_key, {})
            item["confusion"][expected_key][predicted_key] = item["confusion"][expected_key].get(predicted_key, 0) + 1
    accuracies = []
    for component in BASE_COMPONENTS:
        item = by_component[component]
        count = item["count"]
        item["accuracy"] = None if count == 0 else item["correct"] / count
        item["prediction_distribution"] = dict(sorted(item["prediction_distribution"].items()))
        item["confusion"] = {truth: dict(sorted(predictions.items())) for truth, predictions in sorted(item["confusion"].items())}
        if item["accuracy"] is not None:
            accuracies.append(item["accuracy"])
    return {
        "macro_accuracy": None if not accuracies else sum(accuracies) / len(accuracies),
        "by_component": by_component,
    }


@dataclass(frozen=True, slots=True)
class IRMemMetadata:
    version: str
    format_version: str
    model_path: str
    input_dim: int
    hidden_size: int
    components: tuple[str, ...]
    validation_macro_tfsm: float
    encoder_fingerprint: str
    public_train_input_sha256: str
    public_validation_input_sha256: str
    transition_spec_sha256: str
    loss_version: str
    place_kind_loss_weight: float
    pointer_loss_weight: float
    label_smoothing: float
    key_dim: int
    lr_scheduler_patience: int
    scheduler_enabled: bool
    scheduler_factor: float
    seed: int
    initial_learning_rate: float
    epochs: int
    batch_size: int
    grad_clip_norm: float
    representation: str = "structured"

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "IRMemMetadata":
        required = {
            "version", "format_version", "model_path", "input_dim", "hidden_size", "components",
            "validation_macro_tfsm", "encoder_fingerprint", "public_train_input_sha256",
            "public_validation_input_sha256", "transition_spec_sha256", "loss_version",
            "place_kind_loss_weight", "pointer_loss_weight", "label_smoothing", "key_dim",
            "lr_scheduler_patience", "scheduler_enabled", "scheduler_factor",
            "seed", "initial_learning_rate", "epochs", "batch_size", "grad_clip_norm",
            "representation",
        }
        if set(value) != required or not isinstance(value.get("components"), (list, tuple)):
            raise RuntimeError("IRMem checkpoint metadata has an unsupported schema")
        return cls(
            version=str(value["version"]),
            format_version=str(value["format_version"]),
            model_path=str(value["model_path"]),
            input_dim=int(value["input_dim"]),
            hidden_size=int(value["hidden_size"]),
            components=tuple(str(item) for item in value["components"]),
            validation_macro_tfsm=float(value["validation_macro_tfsm"]),
            encoder_fingerprint=str(value["encoder_fingerprint"]),
            public_train_input_sha256=str(value["public_train_input_sha256"]),
            public_validation_input_sha256=str(value["public_validation_input_sha256"]),
            transition_spec_sha256=str(value["transition_spec_sha256"]),
            loss_version=str(value["loss_version"]),
            place_kind_loss_weight=float(value["place_kind_loss_weight"]),
            pointer_loss_weight=float(value["pointer_loss_weight"]),
            label_smoothing=float(value["label_smoothing"]),
            key_dim=int(value["key_dim"]),
            lr_scheduler_patience=int(value["lr_scheduler_patience"]),
            scheduler_enabled=bool(value["scheduler_enabled"]),
            scheduler_factor=float(value["scheduler_factor"]),
            seed=int(value["seed"]),
            initial_learning_rate=float(value["initial_learning_rate"]),
            epochs=int(value["epochs"]),
            batch_size=int(value["batch_size"]),
            grad_clip_norm=float(value["grad_clip_norm"]),
            representation=str(value["representation"]),
        )


def _checkpoint_paths(output_path: str | Path) -> tuple[Path, Path, Path, Path]:
    tensor_path = Path(output_path)
    if tensor_path.suffix != ".safetensors":
        tensor_path = tensor_path.with_suffix(".safetensors")
    stem = tensor_path.with_suffix("")
    return (
        tensor_path,
        stem.with_suffix(".metadata.json"),
        stem.with_suffix(".history.json"),
        stem.with_suffix(".training_report.json"),
    )


def _save_checkpoint(*, output_path: str | Path, metadata: IRMemMetadata, state_dict: Mapping[str, Any], history: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    tensor_path, metadata_path, history_path, report_path = _checkpoint_paths(output_path)
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    _load_file, save_file = _safe_tensor_io()
    save_file(
        {key: value.detach().cpu().contiguous() for key, value in state_dict.items()},
        str(tensor_path),
        metadata={"format_version": SAFE_FORMAT_VERSION, "kind": "irmem_checkpoint"},
    )
    _atomic_json(metadata_path, asdict(metadata))
    _atomic_json(history_path, list(history))
    return {
        "checkpoint": str(tensor_path),
        "metadata": str(metadata_path),
        "history": str(history_path),
        "training_report": str(report_path),
    }


def _load_checkpoint(checkpoint_path: str | Path) -> tuple[IRMemMetadata, Mapping[str, Any]]:
    tensor_path, metadata_path, _history_path, _report_path = _checkpoint_paths(checkpoint_path)
    if not tensor_path.is_file() or not metadata_path.is_file():
        raise RuntimeError("IRMem checkpoint requires paired .safetensors and .metadata.json files")
    try:
        metadata_raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("IRMem checkpoint metadata is unreadable") from exc
    if not isinstance(metadata_raw, Mapping):
        raise RuntimeError("IRMem checkpoint metadata must be an object")
    metadata = IRMemMetadata.from_json(metadata_raw)
    if metadata.version != IRMEM_VERSION or metadata.format_version != SAFE_FORMAT_VERSION:
        raise RuntimeError("unsupported IRMem checkpoint version")
    load_file, _save_file = _safe_tensor_io()
    return metadata, load_file(str(tensor_path), device="cpu")


def train_irmem(
    *, dataset_root: str | Path, model_path: str | Path, output_path: str | Path, cache_root: str | Path,
    device: str = "cuda:0", hidden_size: int = 768, epochs: int = 20, learning_rate: float = 1e-3,
    seed: int = 20260803, representation: str = "structured",
    key_dim: int = KEY_DIM,
    label_smoothing: float = LABEL_SMOOTHING,
    lr_scheduler_patience: int = LR_SCHEDULER_PATIENCE,
) -> dict[str, Any]:
    """Train on public train labels and select on public validation labels."""

    torch, _nn = _torch()
    started = time.monotonic()
    random.seed(seed)
    torch.manual_seed(seed)
    if representation not in REPRESENTATIONS:
        raise RuntimeError(f"IRMem has unsupported representation: {representation}")
    dataset, cache = Path(dataset_root), Path(cache_root)
    model_path = Path(model_path).resolve()
    encoder_fingerprint = _encoder_fingerprint(model_path)
    _reset_device_peak(device)
    cache_started = time.monotonic()

    try:
        embeddings = _embedding_cache(
            dataset_root=dataset,
            model_path=model_path,
            cache_root=cache,
            encoder_fingerprint=encoder_fingerprint,
            representation=representation,
            encoder=None,
        )
        encoder_usage: dict[str, int | float] = {"encoder_token_count": 0, "encoder_forward_count": 0, "encoder_seconds": 0.0}
        cache_status = "hit"
    except RuntimeError as exc:
        if "cache miss requires an encoder" not in str(exc):
            raise
        encoder = FrozenQwenEncoder(model_path=model_path, device=device)
        try:
            embeddings = _embedding_cache(
                dataset_root=dataset,
                model_path=model_path,
                cache_root=cache,
                encoder_fingerprint=encoder_fingerprint,
                representation=representation,
                encoder=encoder,
            )
            encoder_usage = encoder.usage()
            cache_status = "miss_built"
            encoder_phase_memory = _device_memory(device)
        finally:
            encoder.close()
    if cache_status == "hit":
        encoder_phase_memory = _device_memory(device)
    cache_seconds = time.monotonic() - cache_started
    if embeddings is None:
        raise RuntimeError("IRMem failed to obtain public embeddings")

    train_labels, val_labels = _labels(dataset, split="train"), _labels(dataset, split="validation")
    train = [(example, train_labels[example.query_id]) for example in load_examples(dataset, split="train", representation=representation)]
    validation = [(example, val_labels[example.query_id]) for example in load_examples(dataset, split="validation", representation=representation)]
    _reset_device_peak(device)
    network = _network(embeddings.input_dim, hidden_size, key_dim=key_dim).to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=learning_rate)

    scheduler_enabled = lr_scheduler_patience > 0
    scheduler = None
    if scheduler_enabled:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=LR_SCHEDULER_FACTOR,
            patience=lr_scheduler_patience, verbose=False,
        )
    training_started = time.monotonic()
    best, best_state, best_epoch, history = -1.0, None, None, []
    final_learning_rate = learning_rate
    for epoch in range(1, epochs + 1):
        network.train()
        random.shuffle(train)
        losses = _train_epoch(network, optimizer, train, embeddings, device=device, label_smoothing=label_smoothing)
        network.eval()
        train_metrics = _prediction_diagnostics(network, train, embeddings, device=device)
        validation_metrics = _prediction_diagnostics(network, validation, embeddings, device=device)
        macro = validation_metrics["macro_accuracy"]
        if macro is None:
            raise RuntimeError("IRMem validation has no component metrics")

        if scheduler is not None:
            scheduler.step(macro)
        current_lr = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": sum(losses) / len(losses),
                "validation_macro_tfsm": macro,
                "learning_rate": current_lr,
                "train_diagnostics": train_metrics,
                "validation_diagnostics": validation_metrics,
            }
        )
        if macro > best:
            best, best_state, best_epoch = macro, {key: value.detach().cpu().clone() for key, value in network.state_dict().items()}, epoch
        final_learning_rate = current_lr
    training_seconds = time.monotonic() - training_started
    if best_state is None:
        raise RuntimeError("IRMem did not produce a checkpoint")
    input_hashes = _public_input_fingerprints(dataset, splits=TRAINING_SPLITS, representation=representation)
    metadata = IRMemMetadata(
        version=IRMEM_VERSION,
        format_version=SAFE_FORMAT_VERSION,
        model_path=str(model_path),
        input_dim=embeddings.input_dim,
        hidden_size=hidden_size,
        components=tuple(BASE_COMPONENTS),
        validation_macro_tfsm=best,
        encoder_fingerprint=encoder_fingerprint,
        public_train_input_sha256=input_hashes["train"],
        public_validation_input_sha256=input_hashes["validation"],
        transition_spec_sha256=spec_provenance(representation)["sha256"],
        loss_version=LOSS_VERSION,
        place_kind_loss_weight=PLACE_KIND_LOSS_WEIGHT,
        pointer_loss_weight=POINTER_LOSS_WEIGHT,
        label_smoothing=label_smoothing,
        key_dim=key_dim,
        lr_scheduler_patience=lr_scheduler_patience,
        scheduler_enabled=scheduler_enabled,
        scheduler_factor=LR_SCHEDULER_FACTOR,
        seed=seed,
        initial_learning_rate=learning_rate,
        epochs=epochs,
        batch_size=TRAIN_BATCH_SIZE,
        grad_clip_norm=GRAD_CLIP_NORM,
        representation=representation,
    )
    paths = _save_checkpoint(output_path=output_path, metadata=metadata, state_dict=best_state, history=history)
    training_report = {
        "status": "completed",
        "irmem_version": IRMEM_VERSION,
        "representation": representation,
        "checkpoint": paths["checkpoint"],
        "metadata": paths["metadata"],
        "history": paths["history"],
        "cache_status": cache_status,
        "cache_seconds": round(cache_seconds, 6),
        "cache_encoder_usage": encoder_usage,
        "cache_encoder_peak_gpu_memory": encoder_phase_memory,
        "gru_training_seconds": round(training_seconds, 6),
        "gru_training_gpu_hours": round(training_seconds / 3600.0 if device.startswith("cuda") else 0.0, 8),
        "gru_training_peak_gpu_memory": _device_memory(device),
        "trainable_parameter_count": int(sum(parameter.numel() for parameter in network.parameters())),
        "best_epoch": best_epoch,
        "best_validation_diagnostics": next(item["validation_diagnostics"] for item in history if item["epoch"] == best_epoch),
        "final_train_diagnostics": history[-1]["train_diagnostics"],
        "final_validation_diagnostics": history[-1]["validation_diagnostics"],
        "seed": seed,
        "initial_learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": TRAIN_BATCH_SIZE,
        "grad_clip_norm": GRAD_CLIP_NORM,
        "final_learning_rate": final_learning_rate,
        "scheduler_enabled": scheduler_enabled,
        "scheduler_patience": lr_scheduler_patience,
        "scheduler_factor": LR_SCHEDULER_FACTOR,
        "test_input_access_during_training": False,
        "test_label_access_during_training": False,
        "metadata_provenance": asdict(metadata),
        "total_wall_seconds": round(time.monotonic() - started, 6),
    }
    _atomic_json(Path(paths["training_report"]), training_report)
    return training_report


class IRMem(MemoryMethod):
    name = "irmem"

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        device: str = "cuda:0",
        representation: str = "structured",
        model_path_override: str | Path | None = None,
    ) -> None:
        """Load an IRMem checkpoint with a locally supplied frozen encoder.

        Anonymous/public checkpoint metadata intentionally uses the sentinel
        ``__LOCAL_MODEL_PATH_REQUIRED__`` in place of the training machine's
        path.  A caller must then pass ``model_path_override``.  The exact
        encoder fingerprint and hidden dimension remain checked, so replacing
        a path cannot silently substitute a different model.
        """
        torch, _nn = _torch()
        metadata, state_dict = _load_checkpoint(checkpoint_path)
        self.metadata = metadata
        self.device = device
        self.representation = representation
        if representation not in REPRESENTATIONS or self.metadata.representation != representation:
            raise RuntimeError("IRMem checkpoint representation does not match requested run")
        if tuple(self.metadata.components) != tuple(BASE_COMPONENTS):
            raise RuntimeError("IRMem checkpoint components do not match E1 Base components")
        if self.metadata.loss_version != LOSS_VERSION:
            raise RuntimeError("IRMem checkpoint uses an unsupported loss protocol")
        if self.metadata.transition_spec_sha256 != spec_provenance(representation)["sha256"]:
            raise RuntimeError("IRMem checkpoint transition-spec provenance mismatch")
        stored_model_path = self.metadata.model_path
        if stored_model_path == "__LOCAL_MODEL_PATH_REQUIRED__":
            if model_path_override is None:
                raise RuntimeError("IRMem checkpoint requires --model-path for its locally supplied frozen encoder")
            model_path = Path(model_path_override).resolve()
        elif model_path_override is None:
            model_path = Path(stored_model_path).resolve()
        else:
            model_path = Path(model_path_override).resolve()
        if _model_dimension(model_path) != self.metadata.input_dim:
            raise RuntimeError("IRMem encoder dimension provenance mismatch")
        if _encoder_fingerprint(model_path) != self.metadata.encoder_fingerprint:
            raise RuntimeError("IRMem encoder weight/tokenizer provenance mismatch")
        self.encoder = FrozenQwenEncoder(model_path=model_path, device=device)
        if self.encoder.dimension != self.metadata.input_dim:
            self.encoder.close()
            raise RuntimeError("IRMem loaded encoder dimension mismatch")
        self.network = _network(self.metadata.input_dim, self.metadata.hidden_size, key_dim=self.metadata.key_dim).to(device).eval()
        self.network.load_state_dict(state_dict)
        self.network.requires_grad_(False)
        self._decoder_forward_count = 0
        self._gru_update_count = 0
        self.reset()

    def reset(self) -> None:
        self.hidden = self.network.initial(device=self.device)
        self.entities = {"agent_1"}
        self.entity_vectors: dict[str, Any] = {}
        self.event_count = 0

    def _entity_vector(self, entity: str):
        if entity not in self.entity_vectors:
            self.entity_vectors[entity] = self.encoder.encode([_entity_text(entity)])[0]
        return self.entity_vectors[entity]

    def step(self, event: PublicEvent) -> Sequence[Generation]:
        with self.encoder.torch.inference_mode():
            event_vector = self.encoder.encode([_event_text(event)])[0].to(self.device)
            self.hidden = self.network.update(event_vector, self.hidden).detach()
        if self.hidden.requires_grad or self.hidden.grad_fn is not None:
            raise RuntimeError("IRMem online hidden state must not retain an autograd graph")
        self.entities.update(_entities((event,)))
        self.event_count += 1
        self._gru_update_count += 1
        return ()

    def _answer(self, *, entity_id: str, component: str) -> Any:
        torch, _nn = _torch()
        self.entities.add(entity_id)
        entities = tuple(sorted(self.entities))
        with torch.inference_mode():
            candidates = torch.stack([self._entity_vector(entity) for entity in entities])
            query = self.encoder.encode([_query_text(entity_id, component)])[0].to(self.device)
            logits = self.network.logits(self.hidden, query, candidates)
        self._decoder_forward_count += 1
        if component == "openness":
            return ("open", "closed")[int(logits["openness"].argmax(dim=-1).item())]
        if component == "toggle_state":
            return ("on", "off")[int(logits["toggle"].argmax(dim=-1).item())]
        return {
            "kind": PLACE_KINDS[int(logits["place_kind"].argmax(dim=-1).item())],
            "target": entities[int(logits["pointer"].argmax(dim=-1).item())],
        }

    def answer(self, example: QueryExample) -> tuple[Any, Sequence[Generation]]:
        return self._answer(entity_id=example.target_entity_id, component=example.target_component), ()

    def dump(self, scope: Sequence[str]) -> tuple[Mapping[str, Any], Sequence[Generation]]:
        return {slot: self._answer(entity_id=slot.rsplit("|", 1)[0], component=slot.rsplit("|", 1)[1]) for slot in scope}, ()

    def compute_usage(self) -> Mapping[str, Any]:
        """Cumulative non-generative online compute; never represented as zero cost."""

        return {
            "llm_generation_call_count": 0,
            **self.encoder.usage(),
            "gru_update_count": self._gru_update_count,
            "decoder_forward_count": self._decoder_forward_count,
            "online_hidden_requires_grad": bool(self.hidden.requires_grad),
            "online_hidden_has_grad_fn": self.hidden.grad_fn is not None,
        }

    def trace(self) -> Mapping[str, Any]:
        return {
            "method": self.name,
            "representation": self.representation,
            "event_count": self.event_count,
            "entity_count": len(self.entities),
            "hidden_size": self.metadata.hidden_size,
            "online_compute": self.compute_usage(),
        }
