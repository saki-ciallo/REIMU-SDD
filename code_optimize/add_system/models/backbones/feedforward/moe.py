from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from fla.modules import GatedMLP
from torch import nn

from .experts import SwiGLUExperts
from .router import TopKRouter


@dataclass(slots=True)
class MoEOutput:
    hidden_states: torch.Tensor
    aux_loss: torch.Tensor | None = None


def zero_aux_loss(reference: torch.Tensor) -> torch.Tensor:
    return torch.zeros((), device=reference.device, dtype=torch.float32)


class LatentMoE(nn.Module):
    """Top-k routed latent SwiGLU experts with optional shared dense expert."""

    def __init__(
        self,
        *,
        hidden_size: int,
        latent_hidden_size: int,
        latent_intermediate_size: int,
        shared_intermediate_size: int | None,
        num_experts: int,
        top_k: int,
        bias: bool,
        capacity_factor: float | None,
        drop_tokens: bool,
        aux_loss_coefficient: float,
        use_sequence_aux_loss: bool,
        use_shared_expert: bool,
        return_aux_loss: bool,
        mlp_hidden_ratio: float | None,
        mlp_hidden_act: str,
        mlp_fuse_swiglu: bool,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = 1 if num_experts == 1 else top_k
        self.capacity_factor = capacity_factor
        self.drop_tokens = drop_tokens
        self.aux_loss_coefficient = aux_loss_coefficient
        self.use_sequence_aux_loss = use_sequence_aux_loss
        self.return_aux_loss = return_aux_loss

        if num_experts == 1:
            self.router = None
            self.input_projection = nn.Identity()
            self.experts: nn.Module = GatedMLP(
                hidden_size=hidden_size,
                hidden_ratio=mlp_hidden_ratio,
                intermediate_size=shared_intermediate_size,
                hidden_act=mlp_hidden_act,
                fuse_swiglu=mlp_fuse_swiglu,
            )
            self.output_projection = nn.Identity()
            self.shared_expert = None
            return

        self.router = TopKRouter(hidden_size, num_experts, self.top_k)
        self.input_projection = (
            nn.Identity()
            if latent_hidden_size == hidden_size
            else nn.Linear(hidden_size, latent_hidden_size, bias=bias)
        )
        self.experts = SwiGLUExperts(
            latent_hidden_size,
            latent_intermediate_size,
            num_experts,
            bias=bias,
        )
        self.output_projection = (
            nn.Identity()
            if latent_hidden_size == hidden_size
            else nn.Linear(latent_hidden_size, hidden_size, bias=bias)
        )
        self.shared_expert = (
            GatedMLP(
                hidden_size=hidden_size,
                hidden_ratio=mlp_hidden_ratio,
                intermediate_size=shared_intermediate_size,
                hidden_act=mlp_hidden_act,
                fuse_swiglu=mlp_fuse_swiglu,
            )
            if use_shared_expert
            else None
        )

    def _aux_loss(
        self,
        probabilities: torch.Tensor,
        indices: torch.Tensor,
        batch_size: int,
        sequence_length: int,
    ) -> torch.Tensor:
        if self.aux_loss_coefficient == 0.0:
            return zero_aux_loss(probabilities)
        selected = F.one_hot(indices, num_classes=self.num_experts).to(probabilities.dtype)
        if self.use_sequence_aux_loss:
            selected = selected.view(
                batch_size,
                sequence_length,
                self.top_k,
                self.num_experts,
            )
            probabilities = probabilities.view(
                batch_size,
                sequence_length,
                self.num_experts,
            )
            balance = (selected.mean(dim=(1, 2)) * probabilities.mean(dim=1)).sum(dim=-1)
            return self.num_experts * balance.mean() * self.aux_loss_coefficient
        balance = selected.mean(dim=(0, 1)) * probabilities.mean(dim=0)
        return self.num_experts * balance.sum() * self.aux_loss_coefficient

    def _route(
        self,
        indices: torch.Tensor,
        weights: torch.Tensor,
        num_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        token_indices = torch.arange(
            num_tokens,
            device=indices.device,
        ).repeat_interleave(self.top_k)
        expert_indices = indices.flatten()
        pair_weights = weights.flatten()
        order = expert_indices.argsort(stable=True)
        sorted_experts = expert_indices[order]
        sorted_tokens = token_indices[order]
        sorted_weights = pair_weights[order]
        counts = torch.bincount(sorted_experts, minlength=self.num_experts)

        if not self.drop_tokens:
            return sorted_tokens, sorted_weights, counts

        capacity = math.ceil(self.capacity_factor * num_tokens * self.top_k / self.num_experts)
        starts = counts.cumsum(0) - counts
        positions = (
            torch.arange(sorted_experts.numel(), device=indices.device) - starts[sorted_experts]
        )
        keep = positions < capacity
        sorted_experts = sorted_experts[keep]
        sorted_tokens = sorted_tokens[keep]
        sorted_weights = sorted_weights[keep]
        counts = torch.bincount(sorted_experts, minlength=self.num_experts)
        weight_sums = torch.zeros(
            num_tokens,
            device=weights.device,
            dtype=weights.dtype,
        )
        weight_sums.index_add_(0, sorted_tokens, sorted_weights)
        sorted_weights = sorted_weights / weight_sums[sorted_tokens].clamp_min(1e-12)
        return sorted_tokens, sorted_weights, counts

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor | MoEOutput:
        batch_size, sequence_length, hidden_size = hidden_states.shape
        if hidden_size != self.hidden_size:
            raise ValueError(f"Expected hidden_size={self.hidden_size}, got {hidden_size}.")
        if self.num_experts == 1:
            output = self.experts(hidden_states)
            return (
                MoEOutput(output, zero_aux_loss(hidden_states)) if self.return_aux_loss else output
            )

        flat_states = hidden_states.flatten(0, 1)
        probabilities, top_weights, top_indices = self.router(flat_states)
        aux_loss = (
            self._aux_loss(
                probabilities,
                top_indices,
                batch_size,
                sequence_length,
            )
            if self.training and self.return_aux_loss
            else zero_aux_loss(hidden_states)
        )
        latent_states = self.input_projection(flat_states)
        sorted_tokens, sorted_weights, counts = self._route(
            top_indices,
            top_weights,
            flat_states.shape[0],
        )
        expert_output = self.experts(latent_states[sorted_tokens], counts)
        expert_output = expert_output * sorted_weights.to(expert_output.dtype)[:, None]
        mixed = torch.zeros_like(latent_states)
        mixed.index_add_(0, sorted_tokens, expert_output)
        output = self.output_projection(mixed).unflatten(
            0,
            (batch_size, sequence_length),
        )
        if self.shared_expert is not None:
            output = output + self.shared_expert(hidden_states)
        return MoEOutput(output, aux_loss) if self.return_aux_loss else output
