"""Trainable memory mechanisms over a shared frozen public visual frontend."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .features import ACTION_ORDER, FEATURE_DIM


MEMORY_METHODS = ("recent4", "full_history", "implicit_gru")
COMPONENT_ORDER = ("place", "openness", "toggle_state")
VALUE_VOCAB_SIZE = 10
SUBJECT_VOCAB_SIZE = 24


@dataclass(frozen=True, slots=True)
class MemoryModelConfig:
    method: str
    hidden_dim: int = 256
    action_dim: int = 24
    query_dim: int = 48
    transformer_layers: int = 2
    transformer_heads: int = 8
    dropout: float = 0.10
    recent_window: int = 4
    max_positions: int = 800

    def validate(self) -> None:
        if self.method not in MEMORY_METHODS:
            raise ValueError(f"unsupported memory method: {self.method!r}")
        if self.hidden_dim <= 0 or self.action_dim <= 0 or self.query_dim <= 0:
            raise ValueError("model dimensions must be positive")
        if self.hidden_dim % self.transformer_heads:
            raise ValueError("hidden_dim must be divisible by transformer_heads")
        if self.recent_window <= 0 or self.max_positions <= 0:
            raise ValueError("recent_window and max_positions must be positive")


class MemoryStatePredictor(nn.Module):
    """Recent-window, full-history, or recurrent implicit state decoder.

    The model sees only frozen transition embeddings, raw public action IDs,
    and public query identity (target visual type/colour plus component).  It
    has no access to decoded EventFacts, entity IDs as class labels, runtime
    state, private labels, or reveal RGB.
    """

    def __init__(self, config: MemoryModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.action_embedding = nn.Embedding(len(ACTION_ORDER), config.action_dim)
        self.token_projection = nn.Sequential(
            nn.LayerNorm(FEATURE_DIM + config.action_dim),
            nn.Linear(FEATURE_DIM + config.action_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.subject_embedding = nn.Embedding(SUBJECT_VOCAB_SIZE, config.query_dim)
        self.component_embedding = nn.Embedding(len(COMPONENT_ORDER), config.query_dim)
        self.query_projection = nn.Sequential(
            nn.Linear(config.query_dim * 2, config.hidden_dim), nn.GELU(), nn.LayerNorm(config.hidden_dim)
        )
        if config.method == "implicit_gru":
            self.memory = nn.GRU(config.hidden_dim, config.hidden_dim, batch_first=True)
        elif config.method == "full_history":
            self.position_embedding = nn.Embedding(config.max_positions + 1, config.hidden_dim)
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
        else:
            self.memory = nn.Sequential(
                nn.Linear(config.hidden_dim * config.recent_window, config.hidden_dim),
                nn.GELU(),
                nn.LayerNorm(config.hidden_dim),
                nn.Dropout(config.dropout),
            )
        self.decoder = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim), nn.GELU(), nn.LayerNorm(config.hidden_dim), nn.Linear(config.hidden_dim, VALUE_VOCAB_SIZE)
        )

    def _tokens(self, features: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[-1] != FEATURE_DIM:
            raise ValueError("features must be [batch, time, FEATURE_DIM]")
        if actions.shape != features.shape[:2]:
            raise ValueError("actions must align with features")
        return self.token_projection(torch.cat((features, self.action_embedding(actions)), dim=-1))

    def _query(self, subject: torch.Tensor, component: torch.Tensor) -> torch.Tensor:
        if subject.ndim != 1 or component.shape != subject.shape:
            raise ValueError("subject and component must be aligned batch vectors")
        return self.query_projection(torch.cat((self.subject_embedding(subject), self.component_embedding(component)), dim=-1))

    def _memory_vector(self, tokens: torch.Tensor, lengths: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        if lengths.ndim != 1 or lengths.shape[0] != tokens.shape[0] or int(lengths.min()) <= 0:
            raise ValueError("every sample must have a positive sequence length")
        batch, time, _ = tokens.shape
        if self.config.method == "implicit_gru":
            packed = nn.utils.rnn.pack_padded_sequence(tokens, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, hidden = self.memory(packed)
            return hidden[-1]
        if self.config.method == "full_history":



            if time + 1 > self.config.max_positions:
                raise ValueError(f"public prefix length {time} exceeds max_positions={self.config.max_positions - 1}")
            positions = torch.arange(time + 1, device=tokens.device).unsqueeze(0)
            combined = torch.cat((tokens, query.unsqueeze(1)), dim=1)
            padding = positions < time
            padding = padding & (positions >= lengths.unsqueeze(1))
            encoded = self.memory(combined + self.position_embedding(positions), src_key_padding_mask=padding)
            return encoded[:, -1]


        vectors: list[torch.Tensor] = []
        for row, length in enumerate(lengths.tolist()):
            tail = tokens[row, max(0, length - self.config.recent_window):length]
            if tail.shape[0] < self.config.recent_window:
                tail = torch.cat((torch.zeros(self.config.recent_window - tail.shape[0], tail.shape[1], device=tokens.device, dtype=tokens.dtype), tail), dim=0)
            vectors.append(tail.reshape(-1))
        return self.memory(torch.stack(vectors))

    def forward(self, *, features: torch.Tensor, actions: torch.Tensor, lengths: torch.Tensor, subject: torch.Tensor, component: torch.Tensor) -> torch.Tensor:
        query = self._query(subject, component)
        tokens = self._tokens(features, actions)
        memory = self._memory_vector(tokens, lengths, query)
        return self.decoder(torch.cat((memory, query), dim=-1))
