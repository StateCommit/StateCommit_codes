"""Learned full-frame visual transition Grounder.

This module is method-side only.  It receives exactly the public BEFORE/AFTER
RGB pair, the raw action, and the public entity catalog.  It never imports an
Oracle, simulator metadata, or evaluator labels.  Supervised dataset assembly
and optimization live separately in :mod:`learned_grounder_training`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import numpy as np
from PIL import Image
from torch import nn

from .grounder_protocol import GrounderFact, GrounderProtocolError


MODEL_VERSION = "e0.6-full-frame-transition-net/v1"
COLORS = ("blue", "green", "grey", "purple", "red", "yellow")
ENTITY_TYPES = ("ball", "key", "door", "switch")
SUBJECT_KEYS = tuple(f"{entity_type}:{color}" for entity_type in ENTITY_TYPES for color in COLORS)
ZONE_PATTERNS = ("border", "checker", "diagonal", "dots", "solid", "vertical")
VALUE_TOKENS = ("closed", "off", "on", "open", *ZONE_PATTERNS)
SUBJECT_INDEX = {key: index for index, key in enumerate(SUBJECT_KEYS)}
VALUE_INDEX = {key: index for index, key in enumerate(VALUE_TOKENS)}


def visual_subject_key(entry: dict[str, object]) -> str:
    """Return a public type/color identifier for a visible non-zone entity."""

    entity_type, color = entry.get("entity_type"), entry.get("color")
    key = f"{entity_type}:{color}"
    if key not in SUBJECT_INDEX:
        raise GrounderProtocolError(f"unsupported public visual entity: {key!r}")
    return key


def catalog_maps(entity_catalog: Iterable[dict[str, object]]) -> tuple[dict[str, str], dict[str, str]]:
    """Build public visual-key and zone-pattern bindings for one stream."""

    subject_to_id: dict[str, str] = {}
    pattern_to_zone: dict[str, str] = {}
    for entry in entity_catalog:
        entity_id = entry.get("entity_id")
        entity_type = entry.get("entity_type")
        if not isinstance(entity_id, str) or not isinstance(entity_type, str):
            raise GrounderProtocolError("public entity catalog is malformed")
        if entity_type == "zone":
            pattern = entry.get("pattern")
            if not isinstance(pattern, str) or pattern not in ZONE_PATTERNS or pattern in pattern_to_zone:
                raise GrounderProtocolError("public zone catalog is malformed")
            pattern_to_zone[pattern] = entity_id
        else:
            key = visual_subject_key(entry)
            if key in subject_to_id:
                raise GrounderProtocolError(f"non-unique visible entity key: {key}")
            subject_to_id[key] = entity_id
    if not subject_to_id or not pattern_to_zone:
        raise GrounderProtocolError("public entity catalog has no visible entities or zones")
    return subject_to_id, pattern_to_zone


def _allowed_subject_keys(*, action: str, subject_to_id: dict[str, str]) -> tuple[str, ...]:
    if action == "drop":
        allowed_types = {"ball", "key"}
    elif action == "toggle":
        allowed_types = {"door", "switch"}
    else:
        raise GrounderProtocolError(f"unsupported Grounder action: {action!r}")
    keys = tuple(key for key in subject_to_id if key.split(":", 1)[0] in allowed_types)
    if not keys:
        raise GrounderProtocolError(f"catalog has no valid target for action {action!r}")
    return keys


def _argmax_restricted(logits: torch.Tensor, keys: Iterable[str], index: dict[str, int]) -> str:
    candidates = tuple(keys)
    if not candidates:
        raise GrounderProtocolError("empty restricted prediction vocabulary")
    best = max(candidates, key=lambda key: float(logits[index[key]].item()))
    return best


def decode_logits(
    *, subject_logits: torch.Tensor, value_logits: torch.Tensor, action: str, entity_catalog: Iterable[dict[str, object]]
) -> GrounderFact:
    """Compile model logits into one typed fact using only public bindings."""

    if subject_logits.ndim != 1 or subject_logits.numel() != len(SUBJECT_KEYS):
        raise GrounderProtocolError("subject logits have unexpected shape")
    if value_logits.ndim != 1 or value_logits.numel() != len(VALUE_TOKENS):
        raise GrounderProtocolError("value logits have unexpected shape")
    subject_to_id, pattern_to_zone = catalog_maps(entity_catalog)
    subject_key = _argmax_restricted(
        subject_logits, _allowed_subject_keys(action=action, subject_to_id=subject_to_id), SUBJECT_INDEX
    )
    entity_id = subject_to_id[subject_key]
    entity_type = subject_key.split(":", 1)[0]
    if action == "drop":
        pattern = _argmax_restricted(value_logits, pattern_to_zone, VALUE_INDEX)
        return GrounderFact(entity_id, "place", {"kind": "in_zone", "target": pattern_to_zone[pattern]})
    if entity_type == "door":
        state = _argmax_restricted(value_logits, ("open", "closed"), VALUE_INDEX)
        return GrounderFact(entity_id, "openness", state)
    if entity_type == "switch":
        state = _argmax_restricted(value_logits, ("on", "off"), VALUE_INDEX)
        return GrounderFact(entity_id, "toggle_state", state)
    raise GrounderProtocolError("toggle decoder selected an unsupported entity type")


class FullFrameTransitionNet(nn.Module):
    """Small ResNet over a six-channel complete BEFORE/AFTER image pair.

    It has no crop/difference-image input.  Concatenation merely lets its
    learned convolutional filters compare the two raw public images.
    """

    def __init__(self) -> None:
        super().__init__()
        try:
            from torchvision.models import resnet18
        except ImportError as error:
            raise RuntimeError("torchvision is required for the learned Grounder") from error
        backbone = resnet18(weights=None)
        old = backbone.conv1
        backbone.conv1 = nn.Conv2d(6, old.out_channels, kernel_size=old.kernel_size, stride=old.stride, padding=old.padding, bias=False)
        nn.init.kaiming_normal_(backbone.conv1.weight, mode="fan_out", nonlinearity="relu")
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.subject_head = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, len(SUBJECT_KEYS)))
        self.value_head = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, len(VALUE_TOKENS)))

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if images.ndim != 4 or images.shape[1] != 6:
            raise ValueError("expected image pairs with shape [batch, 6, height, width]")
        features = self.backbone(images)
        return self.subject_head(features), self.value_head(features)


def load_rgb_pair(before_path: str | Path, after_path: str | Path, *, image_size: int) -> torch.Tensor:
    """Load two full RGB frames as an un-cropped normalized six-channel tensor."""

    if image_size <= 0:
        raise ValueError("image_size must be positive")
    images: list[torch.Tensor] = []
    for path in (Path(before_path), Path(after_path)):
        with Image.open(path) as image:
            rgb = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
            pixels = torch.from_numpy(np.asarray(rgb).copy()).permute(2, 0, 1).float()
            images.append(pixels.div_(127.5).sub_(1.0))
    return torch.cat(images, dim=0)


@dataclass(frozen=True, slots=True)
class LearnedGrounderConfig:
    image_size: int = 160


class LearnedFullFrameGrounder:
    """Inference wrapper that returns valid EventFacts from public inputs."""

    def __init__(self, *, checkpoint_path: str | Path, device: str = "cuda", image_size: int = 160) -> None:
        from safetensors.torch import load_file


        torch.backends.cudnn.enabled = False
        self.device = torch.device(device)
        self.image_size = image_size
        self.model = FullFrameTransitionNet().to(self.device)
        self.model.load_state_dict(load_file(str(checkpoint_path), device=str(self.device)), strict=True)
        self.model.eval()

    @torch.inference_mode()
    def ground(
        self, *, before_image: str | Path, after_image: str | Path, action: str, entity_catalog: Iterable[dict[str, object]]
    ) -> GrounderFact:
        images = load_rgb_pair(before_image, after_image, image_size=self.image_size).unsqueeze(0).to(self.device)
        subject_logits, value_logits = self.model(images)
        return decode_logits(
            subject_logits=subject_logits[0].cpu(), value_logits=value_logits[0].cpu(), action=action, entity_catalog=entity_catalog
        )
