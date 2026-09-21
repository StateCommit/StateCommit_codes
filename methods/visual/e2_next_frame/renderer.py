from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from e2_memory.features import ACTION_ORDER, FEATURE_DIM


FRAME_METHODS = ("recent4", "full_history", "implicit_gru", "wcm_odometry_residual")
RESIDUAL_METHODS = ("wcm_odometry_residual",)
COMPONENT_ORDER = ("place", "openness", "toggle_state")
VALUE_ORDER = ("closed", "off", "on", "open", "border", "checker", "diagonal", "dots", "solid", "vertical")
INTERFACE_ABLATION_VARIANTS = ("history_only", "state_only", "state_binding", "state_binding_alignment")
DEFAULT_INTERFACE_VARIANT = "state_binding_alignment"


def interface_uses_state(variant: str) -> bool:
    if variant not in INTERFACE_ABLATION_VARIANTS:
        raise ValueError(f"unsupported interface-ablation variant: {variant!r}")
    return variant != "history_only"


def interface_uses_binding(variant: str) -> bool:
    if variant not in INTERFACE_ABLATION_VARIANTS:
        raise ValueError(f"unsupported interface-ablation variant: {variant!r}")
    return variant in {"state_binding", "state_binding_alignment"}


def interface_uses_alignment(variant: str) -> bool:
    if variant not in INTERFACE_ABLATION_VARIANTS:
        raise ValueError(f"unsupported interface-ablation variant: {variant!r}")
    return variant == "state_binding_alignment"


@dataclass(frozen=True, slots=True)
class RendererConfig:
    method: str
    interface_variant: str = DEFAULT_INTERFACE_VARIANT
    query_slot_conditioning: bool = False
    hidden_dim: int = 192
    action_dim: int = 24
    transformer_layers: int = 2
    transformer_heads: int = 6
    recent_window: int = 4
    max_positions: int = 1024
    dropout: float = 0.05

    def validate(self) -> None:
        if self.method not in FRAME_METHODS:
            raise ValueError(f"unsupported next-frame method: {self.method!r}")
        if self.interface_variant not in INTERFACE_ABLATION_VARIANTS:
            raise ValueError(f"unsupported interface-ablation variant: {self.interface_variant!r}")
        if self.method != "wcm_odometry_residual" and self.interface_variant != DEFAULT_INTERFACE_VARIANT:
            raise ValueError("interface ablations are defined only for wcm_odometry_residual")
        if self.query_slot_conditioning and self.method == "wcm_odometry_residual":
            raise ValueError("query-slot conditioning is defined only for history renderers")
        if self.hidden_dim <= 0 or self.action_dim <= 0 or self.recent_window <= 0 or self.max_positions <= 0:
            raise ValueError("renderer dimensions must be positive")
        if self.hidden_dim % self.transformer_heads:
            raise ValueError("hidden_dim must be divisible by transformer_heads")


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(max(1, min(8, out_channels // 8)), out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(max(1, min(8, out_channels // 8)), out_channels),
            nn.GELU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class ConditionalFrameRenderer(nn.Module):
    def __init__(self, config: RendererConfig) -> None:
        super().__init__()
        config.validate()
        if config.method not in {"recent4", "full_history", "implicit_gru"}:
            raise ValueError("ConditionalFrameRenderer supports history baselines only")
        self.config = config
        self.in_block = _ConvBlock(6, 32)
        self.down_1 = _ConvBlock(32, 64, stride=2)
        self.down_2 = _ConvBlock(64, 96, stride=2)
        self.down_3 = _ConvBlock(96, 128, stride=2)
        self.context_to_film = nn.Sequential(nn.LayerNorm(config.hidden_dim), nn.Linear(config.hidden_dim, 256))
        self.up_3 = _ConvBlock(128 + 96, 96)
        self.up_2 = _ConvBlock(96 + 64, 64)
        self.up_1 = _ConvBlock(64 + 32, 32)
        self.output = nn.Conv2d(32, 3, kernel_size=1)
        self.action = nn.Embedding(len(ACTION_ORDER), config.action_dim)
        self.token = nn.Sequential(
            nn.LayerNorm(FEATURE_DIM + config.action_dim),
            nn.Linear(FEATURE_DIM + config.action_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        if config.query_slot_conditioning:
            self.query_subject = nn.Embedding(24, 32)
            self.query_component = nn.Embedding(len(COMPONENT_ORDER), 16)
            self.query_slot = nn.Sequential(
                nn.Linear(32 + 16, config.hidden_dim),
                nn.GELU(),
                nn.LayerNorm(config.hidden_dim),
            )
        if config.method == "recent4":
            self.memory = nn.Sequential(
                nn.Linear(config.hidden_dim * config.recent_window, config.hidden_dim),
                nn.GELU(),
                nn.LayerNorm(config.hidden_dim),
            )
        elif config.method == "implicit_gru":
            self.memory = nn.GRU(config.hidden_dim, config.hidden_dim, batch_first=True)
        else:
            self.position = nn.Embedding(config.max_positions, config.hidden_dim)
            layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.transformer_heads,
                dim_feedforward=config.hidden_dim * 4,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.memory = nn.TransformerEncoder(layer, num_layers=config.transformer_layers, norm=nn.LayerNorm(config.hidden_dim))

    def _history_context(self, *, features: torch.Tensor, actions: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[-1] != FEATURE_DIM or actions.shape != features.shape[:2]:
            raise ValueError("malformed public feature/action history")
        if lengths.ndim != 1 or lengths.shape[0] != features.shape[0] or int(lengths.min()) <= 0:
            raise ValueError("every prefix must contain one or more public transitions")
        tokens = self.token(torch.cat((features, self.action(actions)), dim=-1))
        if self.config.method == "implicit_gru":
            packed = nn.utils.rnn.pack_padded_sequence(tokens, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, hidden = self.memory(packed)
            return hidden[-1]
        if self.config.method == "recent4":
            tails: list[torch.Tensor] = []
            for row, length in enumerate(lengths.tolist()):
                tail = tokens[row, max(0, length - self.config.recent_window):length]
                if tail.shape[0] < self.config.recent_window:
                    padding = torch.zeros(self.config.recent_window - tail.shape[0], tail.shape[1], device=tokens.device)
                    tail = torch.cat((padding, tail), dim=0)
                tails.append(tail.reshape(-1))
            return self.memory(torch.stack(tails))
        if tokens.shape[1] > self.config.max_positions:
            raise ValueError("full-history prefix exceeds frozen maximum length")
        positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
        padding = positions >= lengths.unsqueeze(1)
        encoded = self.memory(tokens + self.position(positions), src_key_padding_mask=padding)
        return encoded[torch.arange(tokens.shape[0], device=tokens.device), lengths - 1]

    def _query_slot_context(self, *, subject: torch.Tensor | None, component: torch.Tensor | None) -> torch.Tensor | None:
        if not self.config.query_slot_conditioning:
            if subject is not None or component is not None:
                raise ValueError("legacy history renderer does not accept query-slot fields")
            return None
        if subject is None or component is None:
            raise ValueError("query-conditioned history renderer requires public entity and component fields")
        return self.query_slot(torch.cat((self.query_subject(subject), self.query_component(component)), dim=1))

    def forward(
        self,
        *,
        current: torch.Tensor,
        binding: torch.Tensor,
        features: torch.Tensor,
        actions: torch.Tensor,
        lengths: torch.Tensor,
        subject: torch.Tensor | None = None,
        component: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if current.ndim != 4 or current.shape[1] != 3 or binding.shape != current.shape:
            raise ValueError("current and binding must be aligned RGB tensors")
        context = self._history_context(features=features, actions=actions, lengths=lengths)
        query_slot = self._query_slot_context(subject=subject, component=component)
        if query_slot is not None:
            context = context + query_slot
        x1 = self.in_block(torch.cat((current, binding), dim=1))
        x2 = self.down_1(x1)
        x3 = self.down_2(x2)
        x4 = self.down_3(x3)
        scale, bias = self.context_to_film(context).chunk(2, dim=1)
        x4 = x4 * (1.0 + scale.unsqueeze(-1).unsqueeze(-1)) + bias.unsqueeze(-1).unsqueeze(-1)
        u3 = self.up_3(torch.cat((nn.functional.interpolate(x4, size=x3.shape[-2:], mode="bilinear", align_corners=False), x3), dim=1))
        u2 = self.up_2(torch.cat((nn.functional.interpolate(u3, size=x2.shape[-2:], mode="bilinear", align_corners=False), x2), dim=1))
        u1 = self.up_1(torch.cat((nn.functional.interpolate(u2, size=x1.shape[-2:], mode="bilinear", align_corners=False), x1), dim=1))
        return torch.tanh(self.output(u1))


class WCMResidualRenderer(nn.Module):
    def __init__(self, *, config: RendererConfig, base_config: RendererConfig) -> None:
        super().__init__()
        if config.method != "wcm_odometry_residual":
            raise ValueError("WCMResidualRenderer requires wcm_odometry_residual")
        if base_config.method != "recent4":
            raise ValueError("the residual renderer requires a Recent-4 base renderer")
        if base_config.query_slot_conditioning:
            raise ValueError("the residual renderer requires an unconditioned frozen Recent-4 base renderer")
        config.validate()
        base_config.validate()
        self.config, self.base_config = config, base_config
        self.base = ConditionalFrameRenderer(base_config)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.subject = nn.Embedding(24, 32)
        self.component = nn.Embedding(len(COMPONENT_ORDER), 16)
        self.value = nn.Embedding(len(VALUE_ORDER), 32)
        self.state = nn.Sequential(
            nn.Linear(32 + 16 + 32 + 1 + 1, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
        )
        self.scene_1 = _ConvBlock(12, 32)
        self.scene_2 = _ConvBlock(32, 64, stride=2)
        self.scene_3 = _ConvBlock(64, 96, stride=2)
        self.scene_4 = _ConvBlock(96, 128, stride=2)
        self.binding_1 = _ConvBlock(3, 32)
        self.binding_2 = _ConvBlock(32, 64, stride=2)
        self.binding_3 = _ConvBlock(64, 96, stride=2)
        self.binding_4 = _ConvBlock(96, 128, stride=2)
        self.query = nn.Conv2d(128, 64, kernel_size=1, bias=False)
        self.key = nn.Conv2d(128, 64, kernel_size=1, bias=False)
        self.val = nn.Conv2d(128, 128, kernel_size=1, bias=False)
        self.state_to_query = nn.Linear(config.hidden_dim, 64, bias=False)
        self.fuse = _ConvBlock(256, 128)
        self.context_to_film = nn.Sequential(nn.LayerNorm(config.hidden_dim), nn.Linear(config.hidden_dim, 256))
        self.up_3 = _ConvBlock(128 + 96, 96)
        self.up_2 = _ConvBlock(96 + 64, 64)
        self.up_1 = _ConvBlock(64 + 32, 32)
        self.delta = nn.Conv2d(32, 3, kernel_size=1)
        self.gate = nn.Conv2d(32, 1, kernel_size=1)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -4.0)

    def train(self, mode: bool = True) -> "WCMResidualRenderer":
        super().train(mode)
        self.base.eval()
        return self

    def load_frozen_base(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.base.load_state_dict(state_dict, strict=True)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.base.eval()

    def _state_context(
        self,
        *,
        subject: torch.Tensor | None,
        component: torch.Tensor | None,
        value: torch.Tensor | None,
        binding_present: torch.Tensor | None,
        projection_present: torch.Tensor | None,
    ) -> torch.Tensor:
        if any(item is None for item in (subject, component, value, binding_present, projection_present)):
            raise ValueError("odometry residual renderer requires compiled query-state fields")
        assert subject is not None and component is not None and value is not None
        assert binding_present is not None and projection_present is not None
        return self.state(
            torch.cat(
                (
                    self.subject(subject),
                    self.component(component),
                    self.value(value),
                    binding_present.float().unsqueeze(1),
                    projection_present.float().unsqueeze(1),
                ),
                dim=1,
            )
        )

    def _spatial_retrieve(self, *, scene: torch.Tensor, binding: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = scene.shape
        query = self.query(scene).flatten(2).transpose(1, 2)
        query = query + self.state_to_query(state).unsqueeze(1)
        key = self.key(binding).flatten(2)
        value = self.val(binding).flatten(2).transpose(1, 2)
        attention = torch.softmax(query @ key / (query.shape[-1] ** 0.5), dim=-1)
        retrieved = attention @ value
        return retrieved.transpose(1, 2).reshape(batch, 128, height, width)

    def forward(
        self,
        *,
        current: torch.Tensor,
        binding: torch.Tensor,
        features: torch.Tensor,
        actions: torch.Tensor,
        lengths: torch.Tensor,
        subject: torch.Tensor | None = None,
        component: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        binding_present: torch.Tensor | None = None,
        binding_projection: torch.Tensor | None = None,
        projection_present: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if current.ndim != 4 or current.shape[1] != 3 or binding.shape != current.shape:
            raise ValueError("current and binding must be aligned RGB tensors")
        if binding_projection is None or binding_projection.shape != current.shape:
            raise ValueError("odometry residual renderer requires an aligned public binding projection")
        state = self._state_context(
            subject=subject,
            component=component,
            value=value,
            binding_present=binding_present,
            projection_present=projection_present,
        )
        with torch.no_grad():
            base = self.base(current=current, binding=torch.zeros_like(current), features=features, actions=actions, lengths=lengths)
        x1 = self.scene_1(torch.cat((current, base, binding, binding_projection), dim=1))
        x2 = self.scene_2(x1)
        x3 = self.scene_3(x2)
        scene = self.scene_4(x3)
        b1 = self.binding_1(binding)
        b2 = self.binding_2(b1)
        b3 = self.binding_3(b2)
        bound = self.binding_4(b3)
        x4 = self.fuse(torch.cat((scene, self._spatial_retrieve(scene=scene, binding=bound, state=state)), dim=1))
        scale, bias = self.context_to_film(state).chunk(2, dim=1)
        x4 = x4 * (1.0 + scale.unsqueeze(-1).unsqueeze(-1)) + bias.unsqueeze(-1).unsqueeze(-1)
        u3 = self.up_3(torch.cat((nn.functional.interpolate(x4, size=x3.shape[-2:], mode="bilinear", align_corners=False), x3), dim=1))
        u2 = self.up_2(torch.cat((nn.functional.interpolate(u3, size=x2.shape[-2:], mode="bilinear", align_corners=False), x2), dim=1))
        u1 = self.up_1(torch.cat((nn.functional.interpolate(u2, size=x1.shape[-2:], mode="bilinear", align_corners=False), x1), dim=1))
        return torch.clamp(base + torch.sigmoid(self.gate(u1)) * torch.tanh(self.delta(u1)), min=-1.0, max=1.0)


def build_renderer(*, config: RendererConfig, base_config: RendererConfig | None = None) -> nn.Module:
    if config.method == "wcm_odometry_residual":
        if base_config is None:
            raise ValueError("residual renderer requires its frozen Recent-4 base config")
        return WCMResidualRenderer(config=config, base_config=base_config)
    if base_config is not None:
        raise ValueError("history renderers do not accept a frozen base config")
    return ConditionalFrameRenderer(config)
