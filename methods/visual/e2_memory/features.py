"""Public-only frozen visual feature extraction for memory baselines.

Each transition feature is obtained from the *backbone only* of E0.6's
full-frame learned Grounder.  The semantic heads, entity catalog bindings,
and private EventFact labels are never used here.  This gives every baseline
the same visual transition frontend while leaving the memory mechanism as the
only varying component.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from safetensors.torch import load_file, save_file

from e0.public_protocol import PublicBenchmark, PublicStream
from e0_5.learned_grounder import FullFrameTransitionNet, load_rgb_pair


FEATURE_PROTOCOL_VERSION = "e2-memory-public-transition-features/v1"
ACTION_ORDER = ("left", "right", "forward", "pickup", "drop", "toggle")
ACTION_INDEX = {action: index for index, action in enumerate(ACTION_ORDER)}
FEATURE_DIM = 512


class FeatureProtocolError(ValueError):
    """Raised when a public feature cache does not match its declared inputs."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def public_index_sha256(public_root: str | Path) -> str:
    """Fingerprint only the public index documents, never evaluator records."""

    root = Path(public_root)
    digest = hashlib.sha256()
    for name in ("streams.json", "queries.json", "entity_catalogs.json"):
        digest.update(name.encode("utf-8"))
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class FeatureCacheManifest:
    protocol_version: str
    split: str
    image_size: int
    checkpoint_sha256: str
    public_index_sha256: str
    stream_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol_version": self.protocol_version,
            "split": self.split,
            "image_size": self.image_size,
            "checkpoint_sha256": self.checkpoint_sha256,
            "public_index_sha256": self.public_index_sha256,
            "stream_ids": list(self.stream_ids),
        }

    @classmethod
    def from_path(cls, path: Path) -> "FeatureCacheManifest":
        data = _read_json(path)
        if not isinstance(data, dict):
            raise FeatureProtocolError("feature manifest must be a JSON object")
        stream_ids = data.get("stream_ids")
        if not isinstance(stream_ids, list) or not all(isinstance(item, str) for item in stream_ids):
            raise FeatureProtocolError("feature manifest stream_ids is malformed")
        return cls(
            protocol_version=str(data.get("protocol_version")),
            split=str(data.get("split")),
            image_size=int(data.get("image_size")),
            checkpoint_sha256=str(data.get("checkpoint_sha256")),
            public_index_sha256=str(data.get("public_index_sha256")),
            stream_ids=tuple(stream_ids),
        )


class FrozenTransitionEncoder:
    """E0.6 backbone as a feature-only, immutable public observation encoder."""

    def __init__(self, *, checkpoint_path: str | Path, device: str, image_size: int) -> None:
        torch.backends.cudnn.enabled = False
        self.device = torch.device(device)
        self.image_size = image_size
        model = FullFrameTransitionNet()
        model.load_state_dict(load_file(str(checkpoint_path), device="cpu"), strict=True)
        self.backbone = model.backbone.to(self.device).eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

    @torch.inference_mode()
    def encode_batch(self, image_pairs: torch.Tensor) -> torch.Tensor:
        if image_pairs.ndim != 4 or image_pairs.shape[1] != 6:
            raise FeatureProtocolError("expected [batch, 6, H, W] image pairs")
        return self.backbone(image_pairs.to(self.device, non_blocking=True)).detach().cpu()


def _stream_action_tensor(stream: PublicStream) -> torch.Tensor:
    actions: list[int] = []
    for frame in stream.frames[1:]:
        if frame.action not in ACTION_INDEX:
            raise FeatureProtocolError(f"public transition has invalid action {frame.action!r}")
        actions.append(ACTION_INDEX[frame.action])
    return torch.tensor(actions, dtype=torch.int64)


def extract_stream_features(
    *, benchmark: PublicBenchmark, stream: PublicStream, encoder: FrozenTransitionEncoder, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode every causal public transition in a stream, in frame order."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    pairs: list[torch.Tensor] = []
    features: list[torch.Tensor] = []
    for index in range(1, len(stream.frames)):
        pairs.append(
            load_rgb_pair(
                benchmark.public_root / stream.frames[index - 1].rgb_path,
                benchmark.public_root / stream.frames[index].rgb_path,
                image_size=encoder.image_size,
            )
        )
        if len(pairs) == batch_size or index == len(stream.frames) - 1:
            features.append(encoder.encode_batch(torch.stack(pairs)))
            pairs.clear()
    transition_features = torch.cat(features, dim=0).to(dtype=torch.float16).contiguous()
    actions = _stream_action_tensor(stream)
    if transition_features.shape != (len(stream.frames) - 1, FEATURE_DIM) or actions.shape[0] != transition_features.shape[0]:
        raise FeatureProtocolError("extracted transition feature shape does not match public frames")
    return transition_features, actions


def extract_split_features(
    *, benchmark: PublicBenchmark, split: str, checkpoint_path: str | Path, output_root: str | Path,
    device: str, image_size: int = 160, batch_size: int = 64
) -> dict[str, object]:
    """Build one immutable public-only feature shard for a single split.

    The caller must invoke this separately for test *after* freezing a model
    configuration.  This function itself never imports a label loader.
    """

    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    root = Path(output_root).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"output root is not empty: {root}")
    root.mkdir(parents=True, exist_ok=False)
    stream_ids = tuple(sorted(stream.stream_id for stream in benchmark.streams.values() if stream.split == split))
    if not stream_ids:
        raise FeatureProtocolError(f"public benchmark has no {split} streams")
    manifest = FeatureCacheManifest(
        protocol_version=FEATURE_PROTOCOL_VERSION,
        split=split,
        image_size=image_size,
        checkpoint_sha256=sha256_file(checkpoint_path),
        public_index_sha256=public_index_sha256(benchmark.public_root),
        stream_ids=stream_ids,
    )
    _atomic_json(root / "manifest.json", manifest.to_dict())
    encoder = FrozenTransitionEncoder(checkpoint_path=checkpoint_path, device=device, image_size=image_size)
    for ordinal, stream_id in enumerate(stream_ids, start=1):
        stream = benchmark.streams[stream_id]
        transition_features, actions = extract_stream_features(
            benchmark=benchmark, stream=stream, encoder=encoder, batch_size=batch_size
        )
        save_file({"transition_features": transition_features, "actions": actions}, str(root / f"{stream_id}.safetensors"))
        _atomic_json(root / "progress.json", {
            "status": "running", "split": split, "completed_streams": ordinal, "total_streams": len(stream_ids),
            "public_only": True,
        })
    _atomic_json(root / "progress.json", {
        "status": "public_feature_cache_complete", "split": split, "completed_streams": len(stream_ids),
        "total_streams": len(stream_ids), "public_only": True,
    })
    return {"status": "passed", "output_root": str(root), "manifest": manifest.to_dict()}


def load_feature_cache(*, root: str | Path, benchmark: PublicBenchmark, split: str, checkpoint_path: str | Path, image_size: int) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Validate and load a frozen public feature shard into CPU memory."""

    directory = Path(root).resolve()
    manifest = FeatureCacheManifest.from_path(directory / "manifest.json")
    expected_ids = tuple(sorted(stream.stream_id for stream in benchmark.streams.values() if stream.split == split))
    expected = FeatureCacheManifest(
        protocol_version=FEATURE_PROTOCOL_VERSION,
        split=split,
        image_size=image_size,
        checkpoint_sha256=sha256_file(checkpoint_path),
        public_index_sha256=public_index_sha256(benchmark.public_root),
        stream_ids=expected_ids,
    )
    if manifest != expected:
        raise FeatureProtocolError("feature cache manifest does not match the frozen public inputs/checkpoint")
    cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for stream_id in expected_ids:
        tensors = load_file(str(directory / f"{stream_id}.safetensors"), device="cpu")
        features, actions = tensors.get("transition_features"), tensors.get("actions")
        if features is None or actions is None or features.ndim != 2 or features.shape[1] != FEATURE_DIM or actions.ndim != 1:
            raise FeatureProtocolError(f"malformed feature shard for {stream_id}")
        if features.shape[0] != len(benchmark.streams[stream_id].frames) - 1 or actions.shape[0] != features.shape[0]:
            raise FeatureProtocolError(f"feature length does not match public stream {stream_id}")
        cache[stream_id] = (features.float(), actions.long())
    return cache
