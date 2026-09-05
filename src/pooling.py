from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
import torch.nn.functional as F

from fla.modules import RMSNorm

from .configuration import GatedAttentionPoolingConfig

POOLING_REGISTRY: dict[str, type[nn.Module]] = {}
POOLING_TYPE_ALIASES = {
    "global_pool": "gamp",
    "gated_attention": "gapv1",
    "gated_attention_v2": "gapv2",
}


def register_pooling(
    pooling_type: str,
) -> Callable[[type[nn.Module]], type[nn.Module]]:
    """Register a pooling implementation under a case-insensitive config name."""

    normalized_type = pooling_type.strip().lower()
    if not normalized_type:
        raise ValueError("pooling_type must be a non-empty string.")

    def decorator(pooling_cls: type[nn.Module]) -> type[nn.Module]:
        if normalized_type in POOLING_REGISTRY:
            raise ValueError(f"pooling_type={normalized_type!r} is already registered.")
        POOLING_REGISTRY[normalized_type] = pooling_cls
        return pooling_cls

    return decorator


def build_pooling(config: GatedAttentionPoolingConfig) -> nn.Module:
    """Build the pooling implementation selected by config.pooling_type."""

    pooling_type = POOLING_TYPE_ALIASES.get(config.pooling_type, config.pooling_type)
    pooling_cls = POOLING_REGISTRY.get(pooling_type)
    if pooling_cls is None:
        supported = ", ".join(sorted(POOLING_REGISTRY)) or "none"
        raise ValueError(
            f"Unsupported pooling_type={config.pooling_type!r}. "
            f"Registered pooling types: {supported}."
        )
    return pooling_cls(config)


def _validate_pooling_input(hidden_states: torch.Tensor, input_size: int) -> None:
    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must have shape [batch_size, seq_len, input_size].")
    if hidden_states.shape[-1] != input_size:
        raise ValueError(
            f"hidden_states last dimension must match input_size={input_size}, "
            f"got {hidden_states.shape[-1]}."
        )


def _build_output_norm(config: GatedAttentionPoolingConfig) -> nn.Module:
    return (
        RMSNorm(config.output_size, eps=config.layer_norm_eps)
        if config.use_output_norm
        else nn.Identity()
    )


def _build_output_act(config: GatedAttentionPoolingConfig) -> nn.Module:
    if config.output_act == "none":
        return nn.Identity()
    if config.output_act == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported output_act={config.output_act!r}.")


@register_pooling("mhgap")
class MultiHeadGatedAttentionPooling(nn.Module):
    """
    Multi-head gated attention pooling.

    Input:
        hidden_states: [B, S, H]

    Output:
        pooled_output: [B, output_size]
    """

    def __init__(self, config: GatedAttentionPoolingConfig) -> None:
        super().__init__()

        self.input_size = config.input_size
        self.output_size = config.output_size
        self.num_heads = config.resolved_num_heads # 检查过值的
        self.head_dim = self.output_size // self.num_heads
        self.gate_act = config.gate_act
        self.score_temperature = config.score_temperature
        self.use_output_norm = config.use_output_norm
        # self.value_proj = nn.Linear(self.input_size, self.output_size, bias=config.bias)
        # self.gate_proj = nn.Linear(self.input_size, self.output_size, bias=config.bias)
        self.value_gate_proj = nn.Linear(self.input_size, 2 * self.output_size, bias=config.bias)
        self.score_weight = nn.Parameter(
            torch.empty(self.num_heads, self.head_dim)
        )
        self.output_act = _build_output_act(config)
        self.output_norm = _build_output_norm(config)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()

        if self.gate_act == "sigmoid":
            self.act = torch.sigmoid
            # self.evidence_norm = nn.Identity()
        elif self.gate_act == "silu":
            self.act = F.silu
            # self.evidence_norm = RMSNorm(self.head_dim, eps=config.layer_norm_eps)
        else:
            raise ValueError(f"Unsupported gate_act={self.gate_act!r}.")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        _validate_pooling_input(hidden_states, self.input_size)
        B, S, _ = hidden_states.shape

        value_gate = self.value_gate_proj(hidden_states)
        values, gates = value_gate.chunk(2, dim=-1)
        values = values.view(B, S, self.num_heads, self.head_dim) # [B, S, input_size] -> [B, S, num_heads, head_dim]
        gates = gates.view(B, S, self.num_heads, self.head_dim) # [B, S, input_size] -> [B, S, K, D]
        # evidence = values * evidence_gate # 多头，逐 token 的表示
        evidence = values * self.act(gates)
        logits = torch.einsum("bshd,hd->bsh", values, self.score_weight)
        attention_weights = torch.softmax((logits / self.score_temperature).float(), dim=1).to(evidence.dtype)
        pooled = torch.einsum("bsh,bshd->bhd", attention_weights, evidence) # 聚合 [B, K, D]
        pooled = pooled.reshape(B, self.output_size)
        return self.dropout(self.output_norm(self.output_act(pooled)))


@register_pooling("gamp")
class GlobalAvgMaxPooling(nn.Module):
    """Concatenate global mean/max features, then project to output_size."""

    def __init__(self, config: GatedAttentionPoolingConfig) -> None:
        super().__init__()
        self.input_size = config.input_size
        self.output_size = config.output_size
        self.fused_proj = nn.Linear(
            2 * self.input_size,
            self.output_size,
            bias=config.bias,
        )
        self.use_output_norm = config.use_output_norm
        self.output_act = _build_output_act(config)
        self.output_norm = _build_output_norm(config)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        _validate_pooling_input(hidden_states, self.input_size)
        avg_pool = hidden_states.mean(dim=1)
        max_pool = hidden_states.amax(dim=1)
        pooled = torch.cat((avg_pool, max_pool), dim=-1)
        pooled = self.fused_proj(pooled)
        return self.dropout(self.output_norm(self.output_act(pooled)))


@register_pooling("gapv1")
class GatedAttentionPooling(nn.Module):
    """Use a SwiGLU scorer to pool the original sequence features."""

    def __init__(self, config: GatedAttentionPoolingConfig) -> None:
        super().__init__()
        self.input_size = config.input_size
        self.output_size = config.output_size
        self.hidden_ratio = config.hidden_ratio
        self.hidden_dim = max(1, int(self.hidden_ratio * self.input_size))
        self.score_temperature = config.score_temperature
        self.gate_value_proj = nn.Linear(
            self.input_size,
            2 * self.hidden_dim,
            bias=False,
        )
        self.to_score = nn.Linear(self.hidden_dim, 1, bias=False)
        self.output_proj = (
            nn.Linear(self.input_size, self.output_size, bias=config.bias)
            if self.input_size != self.output_size
            else nn.Identity()
        )
        self.activation = nn.SiLU()
        self.use_output_norm = config.use_output_norm
        self.output_act = _build_output_act(config)
        self.output_norm = _build_output_norm(config)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        _validate_pooling_input(hidden_states, self.input_size)
        gate, value = self.gate_value_proj(hidden_states).chunk(2, dim=-1)
        score_features = self.activation(gate) * value
        scores = self.to_score(score_features).squeeze(-1)
        weights = torch.softmax(
            (scores / self.score_temperature).float(),
            dim=1,
        ).to(hidden_states.dtype)
        pooled = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)
        pooled = self.output_proj(pooled)
        return self.dropout(self.output_norm(self.output_act(pooled)))


@register_pooling("gapv2")
class GatedAttentionPoolingV2(nn.Module):
    """Use separate SwiGLU scoring and value features for sequence pooling."""

    def __init__(self, config: GatedAttentionPoolingConfig) -> None:
        super().__init__()
        self.input_size = config.input_size
        self.output_size = config.output_size
        self.hidden_ratio = config.hidden_ratio
        self.hidden_dim = max(1, int(self.hidden_ratio * self.input_size))
        self.score_temperature = config.score_temperature
        self.proj = nn.Linear(
            self.input_size,
            3 * self.hidden_dim,
            bias=False,
        )
        self.to_score = nn.Linear(self.hidden_dim, 1, bias=False)
        self.output_proj = nn.Linear(
            self.hidden_dim,
            self.output_size,
            bias=config.bias,
        )
        self.activation = nn.SiLU()
        self.use_output_norm = config.use_output_norm
        self.output_act = _build_output_act(config)
        self.output_norm = _build_output_norm(config)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        _validate_pooling_input(hidden_states, self.input_size)
        gate, score_value, pool_value = self.proj(hidden_states).chunk(3, dim=-1)
        score_features = self.activation(gate) * score_value
        scores = self.to_score(score_features).squeeze(-1)
        weights = torch.softmax(
            (scores / self.score_temperature).float(),
            dim=1,
        ).to(pool_value.dtype)
        pooled = torch.bmm(weights.unsqueeze(1), pool_value).squeeze(1)
        pooled = self.output_proj(pooled)
        return self.dropout(self.output_norm(self.output_act(pooled)))


__all__ = [
    "GatedAttentionPooling",
    "GatedAttentionPoolingV2",
    "GlobalAvgMaxPooling",
    "POOLING_REGISTRY",
    "POOLING_TYPE_ALIASES",
    "MultiHeadGatedAttentionPooling",
    "build_pooling",
    "register_pooling",
]
