from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SwiGLUExperts(nn.Module):
    """BF16 expert weights in the native ``torch.nn.functional.grouped_mm`` layout."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        *,
        bias: bool,
    ) -> None:
        super().__init__()
        factory_kwargs = {"dtype": torch.bfloat16}
        self.num_experts = num_experts
        self.gate_up_projection = nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                2 * intermediate_size,
                **factory_kwargs,
            )
        )
        self.down_projection = nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size,
                hidden_size,
                **factory_kwargs,
            )
        )
        if bias:
            self.gate_up_bias = nn.Parameter(
                torch.empty(
                    num_experts,
                    2 * intermediate_size,
                    **factory_kwargs,
                )
            )
            self.down_bias = nn.Parameter(torch.empty(num_experts, hidden_size, **factory_kwargs))
        else:
            self.register_parameter("gate_up_bias", None)
            self.register_parameter("down_bias", None)

    @staticmethod
    def _grouped_mm(
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        return F.grouped_mm(
            hidden_states.contiguous(),
            weight.contiguous(),
            offs=offsets.contiguous(),
        )

    def _add_grouped_bias(
        self,
        output: torch.Tensor,
        bias: torch.Tensor | None,
        counts: torch.Tensor,
    ) -> torch.Tensor:
        if bias is None:
            return output
        expert_indices = torch.repeat_interleave(
            torch.arange(self.num_experts, device=counts.device),
            counts,
            output_size=output.shape[0],
        )
        return output + bias[expert_indices]

    def forward(
        self,
        hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        counts = tokens_per_expert.to(device=hidden_states.device, dtype=torch.long)
        offsets = counts.cumsum(0).to(torch.int32)
        combined = self._grouped_mm(
            hidden_states,
            self.gate_up_projection,
            offsets,
        )
        combined = self._add_grouped_bias(combined, self.gate_up_bias, counts)
        gates, values = combined.chunk(2, dim=-1)
        activated = F.silu(gates) * values
        output = self._grouped_mm(
            activated,
            self.down_projection,
            offsets,
        )
        return self._add_grouped_bias(output, self.down_bias, counts)
