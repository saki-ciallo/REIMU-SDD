from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

import math

from fla.modules import GatedMLP

@dataclass
class MixerOutput:
    hidden_states: torch.Tensor
    aux_loss: Optional[torch.Tensor] = None



def _new_float32_scalar(reference: torch.Tensor) -> torch.Tensor:
    return torch.zeros((), device=reference.device, dtype=torch.float32)


def _grouped_mm_reference(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    offsets: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Reference implementation for grouped matmul.

    mat_a:   [P, K]
    mat_b:   [E, K, N]
    offsets: [E], cumulative end offsets, int32/int64
    bias:    optional [E, N]

    return:  [P, N]
    """
    outputs = []
    start = 0

    for expert_idx, end_tensor in enumerate(offsets):
        end = int(end_tensor.item())
        if end > start:
            output = mat_a[start:end] @ mat_b[expert_idx]
            if bias is not None:
                output = output + bias[expert_idx]
            outputs.append(output)
        start = end

    out_features = mat_b.shape[-1]

    if not outputs:
        return mat_a.new_empty(mat_a.shape[0], out_features)

    return torch.cat(outputs, dim=0)


@torch.no_grad()
def check_grouped_mm_equivalence(
    *,
    num_experts: int = 4,
    tokens_per_expert: Optional[Tuple[int, ...]] = None,
    in_features: int = 16,
    out_features: int = 32,
    device: Optional[torch.device | str] = None,
    dtype: torch.dtype = torch.bfloat16,
    atol: float = 1e-2,
    rtol: float = 1e-2,
    raise_on_error: bool = True,
) -> bool:
    """
    Forward-only check for grouped_mm against reference matmul.

    For actual training, also run your model's forward + backward once,
    because grouped_mm support can vary across PyTorch versions / devices.
    """
    if num_experts <= 0:
        raise ValueError("num_experts must be positive.")

    if tokens_per_expert is None:
        tokens_per_expert = tuple((idx % 3) + 1 for idx in range(num_experts))

    if len(tokens_per_expert) != num_experts:
        raise ValueError("tokens_per_expert length must equal num_experts.")

    if any(count < 0 for count in tokens_per_expert):
        raise ValueError("tokens_per_expert values must be non-negative.")

    resolved_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    if resolved_device.type != "cuda":
        if raise_on_error:
            raise RuntimeError("F.grouped_mm is expected to run on CUDA.")
        return False


    total_tokens = sum(tokens_per_expert)

    mat_a = torch.randn(
        total_tokens,
        in_features,
        device=resolved_device,
        dtype=dtype,
    )
    mat_b = torch.randn(
        num_experts,
        in_features,
        out_features,
        device=resolved_device,
        dtype=dtype,
    )

    counts = torch.tensor(
        tokens_per_expert,
        device=resolved_device,
        dtype=torch.long,
    )
    offsets = torch.cumsum(counts, dim=0).to(torch.int32)

    try:
        grouped = F.grouped_mm(mat_a, mat_b, offs=offsets)
        reference = _grouped_mm_reference(mat_a, mat_b, offsets)

        torch.testing.assert_close(
            grouped,
            reference,
            atol=atol,
            rtol=rtol,
        )
    except Exception:
        if raise_on_error:
            raise
        return False

    return True


class SwiGLUExperts(nn.Module):
    """
    Grouped SwiGLU experts stored in grouped_mm-native weight layout.

    gate_up_proj: [E, H, 2I]
    down_proj:    [E, I, H]
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        bias: bool,
        dropout: float,
    ) -> None:
        super().__init__()

        if num_experts <= 1:
            raise ValueError("SwiGLUExperts requires num_experts > 1.")

        factory_kwargs = {"dtype": torch.bfloat16} # 必要条件

        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.gate_up_proj = nn.Parameter(
            torch.empty(num_experts, hidden_size, 2 * intermediate_size, **factory_kwargs,)
        )
        self.down_proj = nn.Parameter(
            torch.empty(num_experts, intermediate_size, hidden_size, **factory_kwargs,)
        )

        if bias:
            self.gate_up_bias = nn.Parameter(torch.empty(num_experts, 2 * intermediate_size, **factory_kwargs,)
            )
            self.down_bias = nn.Parameter(torch.empty(num_experts, hidden_size, **factory_kwargs,)
            )
        else:
            self.register_parameter("gate_up_bias", None)
            self.register_parameter("down_bias", None)

    def _grouped_mm(
        self,
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        return F.grouped_mm(
            mat_a.contiguous(),
            mat_b.contiguous(),
            offs=offsets.contiguous(),
        )

    def _expert_indices_from_counts(
        self,
        tokens_per_expert: torch.Tensor,
        output_size: int,
    ) -> torch.Tensor:
        counts = tokens_per_expert.to(dtype=torch.long)

        return torch.repeat_interleave(
            torch.arange(
                self.num_experts,
                device=counts.device,
                dtype=torch.long,
            ),
            counts,
            output_size=int(output_size),
        )

    def _validate_grouped_inputs(
        self,
        hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
    ) -> None:
        if hidden_states.ndim != 2:
            raise ValueError(
                "grouped expert input must have shape [num_pairs, hidden_size]."
            )

        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Expected hidden_size={self.hidden_size}, got {hidden_states.shape[-1]}."
            )

        if tokens_per_expert.ndim != 1 or tokens_per_expert.numel() != self.num_experts:
            raise ValueError(
                f"tokens_per_expert must have shape [{self.num_experts}]."
            )

        if torch.any(tokens_per_expert < 0).item():
            raise ValueError("tokens_per_expert must be non-negative.")

        if int(tokens_per_expert.sum().item()) != hidden_states.shape[0]:
            raise ValueError(
                "sum(tokens_per_expert) must equal hidden_states.shape[0]."
            )

    def _grouped_linear(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        offsets: torch.Tensor,
        expert_indices: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # weight = weight.to(dtype=hidden_states.dtype)
        if bias is not None:
            bias = bias.to(dtype=hidden_states.dtype)
        output = self._grouped_mm(
            hidden_states,
            weight,
            offsets,
        )

        if bias is not None:
            # grouped_mm exposes a bias argument, but current kernels can reject it.
            output = output + bias[expert_indices]

        return output

    def forward_grouped_experts(
        self,
        hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        *,
        validate_shapes: bool = False,
    ) -> torch.Tensor:
        """
        hidden_states:      [P, H], sorted by expert
        tokens_per_expert:  [E], number of rows assigned to each expert

        return:             [P, H]

        Inputs and expert weights are expected to already use a grouped_mm-supported
        dtype, currently BF16 on CUDA.
        """
        if validate_shapes:
            self._validate_grouped_inputs(hidden_states, tokens_per_expert)

        tokens_per_expert = tokens_per_expert.to(
            device=hidden_states.device,
            dtype=torch.long,
        )

        offsets = torch.cumsum(tokens_per_expert, dim=0).to(torch.int32)

        expert_indices = None
        if self.gate_up_bias is not None or self.down_bias is not None:
            expert_indices = self._expert_indices_from_counts(
                tokens_per_expert,
                output_size=hidden_states.shape[0],
            )

        combined = self._grouped_linear(
            hidden_states,
            self.gate_up_proj,
            self.gate_up_bias,
            offsets,
            expert_indices,
        )

        gate, value = combined.chunk(2, dim=-1)
        hidden = self.activation(gate) * value

        output = self._grouped_linear(
            hidden,
            self.down_proj,
            self.down_bias,
            offsets,
            expert_indices,
        )

        return self.dropout(output)

    def forward(
        self,
        hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        *,
        validate_shapes: bool = False,
    ) -> torch.Tensor:
        return self.forward_grouped_experts(
            hidden_states,
            tokens_per_expert,
            validate_shapes=validate_shapes,
        )


class TopKRouter(nn.Module):
    """Top-k router matching the common Qwen-style standalone router shape."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        normalize_topk_prob: bool = True,
    ) -> None:
        super().__init__()
        if num_experts <= 0:
            raise ValueError("num_experts must be positive.")
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        if top_k > num_experts:
            raise ValueError("top_k must be <= num_experts.")
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.normalize_topk_prob = normalize_topk_prob
        self.weight = nn.Parameter(torch.empty(num_experts, hidden_size))

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        router_logits = F.linear(hidden_states, self.weight)
        routing_probs = F.softmax(router_logits.float(), dim=-1)
        top_weights, top_indices = torch.topk(
            routing_probs,
            self.top_k,
            dim=-1,
        )

        if self.normalize_topk_prob:
            top_weights = top_weights / top_weights.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-12)

        return routing_probs, top_weights, top_indices


class LatentMoE(nn.Module):
    """
    Top-k routed LatentMoE with grouped SwiGLU experts.

    - num_experts == 1: ordinary dense GatedMLP using shared_intermediate_size
    - num_experts > 1: latent projection + grouped latent expert computation
    - optional shared expert uses the same GatedMLP sizing as ordinary MLP blocks
    - optional seq/global aux loss
    - optional capacity-based token dropping
    """

    def __init__(
        self,
        hidden_size: int,
        latent_intermediate_size: int,
        num_experts: int,
        latent_hidden_size: Optional[int] = None,
        shared_intermediate_size: Optional[int] = None,
        top_k: int = 2,
        bias: bool = False,
        dropout: float = 0.0,
        capacity_factor: Optional[float] = None,
        drop_tokens: bool = False,
        moe_aux_loss_coeff: float = 0.0,
        use_seq_aux_loss: bool = False,
        use_shared_expert: bool = False,
        return_aux_loss: bool = False,
        mlp_hidden_ratio: Optional[int] = None,
        mlp_hidden_act: str = "swish",
        mlp_fuse_swiglu: bool = True,
    ) -> None:
        super().__init__()

        if num_experts <= 0:
            raise ValueError("num_experts must be positive.")

        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        if num_experts == 1:
            top_k = 1

        if top_k > num_experts:
            raise ValueError("top_k must be <= num_experts.")

        if capacity_factor is not None and capacity_factor <= 0:
            raise ValueError("capacity_factor must be positive or None.")

        if drop_tokens and capacity_factor is None:
            raise ValueError("drop_tokens=True requires capacity_factor.")

        self.hidden_size = hidden_size
        self.latent_hidden_size = latent_hidden_size or hidden_size
        self.latent_intermediate_size = latent_intermediate_size
        self.shared_intermediate_size = shared_intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.bias = bias
        self.capacity_factor = capacity_factor
        self.drop_tokens = drop_tokens
        self.moe_aux_loss_coeff = moe_aux_loss_coeff
        self.use_seq_aux_loss = use_seq_aux_loss
        self.use_shared_expert = use_shared_expert and num_experts > 1
        self.return_aux_loss = return_aux_loss
        self.mlp_hidden_ratio = mlp_hidden_ratio
        self.mlp_hidden_act = mlp_hidden_act
        self.mlp_fuse_swiglu = mlp_fuse_swiglu

        if num_experts == 1:
            self.router = None
            self.down_proj = nn.Identity()
            self.experts = GatedMLP(
                hidden_size=hidden_size,
                hidden_ratio=mlp_hidden_ratio,
                intermediate_size=shared_intermediate_size,
                hidden_act=mlp_hidden_act,
                fuse_swiglu=mlp_fuse_swiglu,
            )
            self.up_proj = nn.Identity()
            self.shared_expert = None
            self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        else:
            self.router = TopKRouter(
                hidden_size=hidden_size,
                num_experts=num_experts,
                top_k=top_k,
            )

            self.down_proj = (
                nn.Identity()
                if self.latent_hidden_size == hidden_size
                else nn.Linear(hidden_size, self.latent_hidden_size, bias=bias)
            )

            self.experts = SwiGLUExperts(
                hidden_size=self.latent_hidden_size,
                intermediate_size=latent_intermediate_size,
                num_experts=num_experts,
                bias=bias,
                dropout=dropout,
            )

            self.up_proj = (
                nn.Identity()
                if self.latent_hidden_size == hidden_size
                else nn.Linear(self.latent_hidden_size, hidden_size, bias=bias)
            )

            self.shared_expert = (
                GatedMLP(
                    hidden_size=hidden_size,
                    hidden_ratio=mlp_hidden_ratio,
                    intermediate_size=shared_intermediate_size,
                    hidden_act=mlp_hidden_act,
                    fuse_swiglu=mlp_fuse_swiglu,
                )
                if self.use_shared_expert
                else None
            )

    def _sequence_aux_loss(
        self,
        routing_probs: torch.Tensor,
        top_indices: torch.Tensor,
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor:
        """
        routing_probs: [B*S, E]
        top_indices:   [B*S, K]

        return: scalar, already multiplied by moe_aux_loss_coeff
        """
        if self.moe_aux_loss_coeff == 0.0:
            return _new_float32_scalar(routing_probs)

        probs = routing_probs.view(batch_size, seq_len, self.num_experts)
        indices = top_indices.view(batch_size, seq_len, self.top_k)

        selected = F.one_hot(
            indices,
            num_classes=self.num_experts,
        ).to(routing_probs.dtype) # [B, S, K, E]

        expert_fraction = selected.mean(dim=(1, 2))  # [B, E]
        expert_prob = probs.mean(dim=1)  # [B, E]

        aux_loss = self.num_experts * torch.sum(
            expert_fraction * expert_prob,
            dim=-1,
        )

        return aux_loss.mean() * self.moe_aux_loss_coeff

    def _global_aux_loss(
        self,
        routing_probs: torch.Tensor,
        top_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        routing_probs: [T, E]
        top_indices:   [T, K]

        return: scalar, already multiplied by moe_aux_loss_coeff
        """
        if self.moe_aux_loss_coeff == 0.0:
            return _new_float32_scalar(routing_probs)

        selected = F.one_hot(
            top_indices,
            num_classes=self.num_experts,
        ).to(routing_probs.dtype) # [T, K, E]

        expert_fraction = selected.mean(dim=(0, 1))  # [E]
        expert_prob = routing_probs.mean(dim=0)  # [E]

        aux_loss = self.num_experts * torch.sum(
            expert_fraction * expert_prob
        )

        return aux_loss * self.moe_aux_loss_coeff

    def _route_tokens(
        self,
        top_indices: torch.Tensor,
        top_weights: torch.Tensor,
        num_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert [T, K] routing result into sorted grouped expert layout.

        return:
            sorted_token_idx:   [P_kept]
            sorted_weight:      [P_kept]
            tokens_per_expert:  [E]
        """
        pair_token_idx = torch.arange(
            num_tokens,
            device=top_indices.device,
        ).repeat_interleave(self.top_k)

        pair_expert_idx = top_indices.reshape(-1)
        pair_weight = top_weights.reshape(-1)

        tokens_per_expert = torch.bincount(
            pair_expert_idx,
            minlength=self.num_experts,
        )

        sort_order = torch.argsort(pair_expert_idx, stable=True)
        sorted_token_idx = pair_token_idx[sort_order]
        sorted_weight = pair_weight[sort_order]

        if self.drop_tokens:
            sorted_expert_idx = pair_expert_idx[sort_order]

            # capacity = torch.ceil(
            #     tokens_per_expert.new_tensor(
            #         self.capacity_factor * num_tokens * self.top_k / self.num_experts,
            #     )
            # ).to(tokens_per_expert.dtype).clamp_min(1)
            capacity = math.ceil(self.capacity_factor * num_tokens * self.top_k / self.num_experts)

            expert_start_offsets = (
                torch.cumsum(tokens_per_expert, dim=0) - tokens_per_expert
            )

            position_in_expert = (
                torch.arange(
                    sorted_expert_idx.numel(),
                    device=sorted_expert_idx.device,
                    dtype=sorted_expert_idx.dtype,
                )
                - expert_start_offsets[sorted_expert_idx]
            )

            keep_mask = position_in_expert < capacity

            sorted_expert_idx = sorted_expert_idx[keep_mask]
            sorted_token_idx = sorted_token_idx[keep_mask]
            sorted_weight = sorted_weight[keep_mask]

            tokens_per_expert = torch.bincount(
                sorted_expert_idx,
                minlength=self.num_experts,
            )

            token_weight_sum = torch.zeros(
                num_tokens,
                device=sorted_weight.device,
                dtype=sorted_weight.dtype,
            )
            token_weight_sum.index_add_(
                0,
                sorted_token_idx,
                sorted_weight,
            )
            sorted_weight = (
                sorted_weight
                / token_weight_sum[sorted_token_idx].clamp_min(1e-12)
            ).to(top_weights.dtype)

        return sorted_token_idx, sorted_weight, tokens_per_expert

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | MixerOutput:
        """
        hidden_states: [B, S, H]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        if hidden_size != self.hidden_size:
            raise ValueError(
                f"Expected hidden_size={self.hidden_size}, got {hidden_size}."
            )

        if self.num_experts == 1:
            output = self.experts(hidden_states)
            output = self.dropout(output)
            if self.return_aux_loss:
                return MixerOutput(
                    hidden_states=output,
                    aux_loss=_new_float32_scalar(hidden_states),
                )

            return output

        num_tokens = batch_size * seq_len

        if num_tokens == 0:
            output = hidden_states.new_empty(
                batch_size,
                seq_len,
                hidden_size,
            )

            if self.return_aux_loss:
                return MixerOutput(
                    hidden_states=output,
                    aux_loss=_new_float32_scalar(hidden_states),
                )

            return output

        flat_states = hidden_states.reshape(num_tokens, hidden_size)

        routing_probs, top_weights, top_indices = self.router(flat_states)

        if self.training and self.return_aux_loss and self.moe_aux_loss_coeff != 0.0:
            if self.use_seq_aux_loss:
                aux_loss = self._sequence_aux_loss(
                    routing_probs=routing_probs,
                    top_indices=top_indices,
                    batch_size=batch_size,
                    seq_len=seq_len,
                )
            else:
                aux_loss = self._global_aux_loss(
                    routing_probs=routing_probs,
                    top_indices=top_indices,
                )
        else:
            aux_loss = _new_float32_scalar(hidden_states)

        flat_latent_states = self.down_proj(flat_states)

        sorted_token_idx, sorted_weight, tokens_per_expert = self._route_tokens(
            top_indices=top_indices,
            top_weights=top_weights,
            num_tokens=num_tokens,
        )

        expert_input = flat_latent_states[sorted_token_idx]

        selected_output = self.experts.forward_grouped_experts(
            expert_input,
            tokens_per_expert=tokens_per_expert,
        )

        selected_output = selected_output * sorted_weight.to(selected_output.dtype).unsqueeze(-1)

        flat_latent_output = torch.zeros_like(flat_latent_states)
        flat_latent_output.index_add_(
            0,
            sorted_token_idx,
            selected_output,
        )
        # weight[token, expert_a] * expert_a(token) + weight[token, expert_b] * expert_b(token) -> flat_latent_output[token]

        flat_output = self.up_proj(flat_latent_output)
        output = flat_output.reshape(batch_size, seq_len, hidden_size)

        if self.shared_expert is not None:
            output = output + self.shared_expert(hidden_states)

        if self.return_aux_loss:
            return MixerOutput(hidden_states=output, aux_loss=aux_loss)

        return output

__all__ = [
    "LatentMoE",
    "MixerOutput",
    "SwiGLUExperts",
    "TopKRouter",
    "check_grouped_mm_equivalence",
]
