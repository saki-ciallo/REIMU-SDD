from __future__ import annotations

import torch
from torch import nn


def pairwise_product(hidden_states: torch.Tensor) -> torch.Tensor:
    nodes = hidden_states[:, :, None, :]
    return nodes * nodes.transpose(1, 2)


class GraphAttentionLayer(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        temperature: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attention_projection = nn.Linear(input_size, output_size)
        self.attention_weight = nn.Parameter(torch.empty(output_size, 1))
        self.attended_projection = nn.Linear(input_size, output_size)
        self.residual_projection = nn.Linear(input_size, output_size)
        self.norm = nn.BatchNorm1d(output_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SELU(inplace=True)
        self.temperature = temperature

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states)
        attention = torch.tanh(self.attention_projection(pairwise_product(hidden_states)))
        attention = attention @ self.attention_weight
        attention = torch.softmax(
            attention.float() / self.temperature,
            dim=-2,
        ).to(hidden_states.dtype)
        attended = attention.squeeze(-1) @ hidden_states
        hidden_states = self.attended_projection(attended) + self.residual_projection(hidden_states)
        shape = hidden_states.shape
        hidden_states = self.norm(hidden_states.flatten(0, 1)).view(shape)
        return self.activation(hidden_states)


class HeterogeneousGraphAttentionLayer(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        temperature: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.type1_projection = nn.Linear(input_size, input_size)
        self.type2_projection = nn.Linear(input_size, input_size)
        self.attention_projection = nn.Linear(input_size, output_size)
        self.master_attention_projection = nn.Linear(input_size, output_size)
        self.attention_weight11 = nn.Parameter(torch.empty(output_size, 1))
        self.attention_weight22 = nn.Parameter(torch.empty(output_size, 1))
        self.attention_weight12 = nn.Parameter(torch.empty(output_size, 1))
        self.master_attention_weight = nn.Parameter(torch.empty(output_size, 1))
        self.attended_projection = nn.Linear(input_size, output_size)
        self.residual_projection = nn.Linear(input_size, output_size)
        self.master_attended_projection = nn.Linear(input_size, output_size)
        self.master_residual_projection = nn.Linear(input_size, output_size)
        self.norm = nn.BatchNorm1d(output_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SELU(inplace=True)
        self.temperature = temperature

    def _attention_map(
        self,
        hidden_states: torch.Tensor,
        num_type1: int,
    ) -> torch.Tensor:
        features = torch.tanh(self.attention_projection(pairwise_product(hidden_states)))
        type11 = features[:, :num_type1, :num_type1] @ self.attention_weight11
        type12 = features[:, :num_type1, num_type1:] @ self.attention_weight12
        type21 = features[:, num_type1:, :num_type1] @ self.attention_weight12
        type22 = features[:, num_type1:, num_type1:] @ self.attention_weight22
        attention = torch.cat(
            (
                torch.cat((type11, type12), dim=2),
                torch.cat((type21, type22), dim=2),
            ),
            dim=1,
        )
        return torch.softmax(
            attention.float() / self.temperature,
            dim=-2,
        ).to(hidden_states.dtype)

    def _update_master(
        self,
        hidden_states: torch.Tensor,
        master: torch.Tensor,
    ) -> torch.Tensor:
        attention = torch.tanh(self.master_attention_projection(hidden_states * master))
        attention = attention @ self.master_attention_weight
        attention = torch.softmax(
            attention.float() / self.temperature,
            dim=-2,
        ).to(hidden_states.dtype)
        attended = attention.squeeze(-1)[:, None, :] @ hidden_states
        return self.master_attended_projection(attended) + self.master_residual_projection(master)

    def forward(
        self,
        type1_states: torch.Tensor,
        type2_states: torch.Tensor,
        master: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_type1 = type1_states.shape[1]
        num_type2 = type2_states.shape[1]
        hidden_states = torch.cat(
            (
                self.type1_projection(type1_states),
                self.type2_projection(type2_states),
            ),
            dim=1,
        )
        hidden_states = self.dropout(hidden_states)
        attention = self._attention_map(hidden_states, num_type1)
        master = self._update_master(hidden_states, master)
        hidden_states = self.attended_projection(
            attention.squeeze(-1) @ hidden_states
        ) + self.residual_projection(hidden_states)
        shape = hidden_states.shape
        hidden_states = self.activation(self.norm(hidden_states.flatten(0, 1)).view(shape))
        return (
            hidden_states[:, :num_type1],
            hidden_states[:, num_type1 : num_type1 + num_type2],
            master,
        )


class GraphPool(nn.Module):
    def __init__(self, ratio: float, input_size: int, dropout: float) -> None:
        super().__init__()
        self.ratio = ratio
        self.score_projection = nn.Linear(input_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        scores = torch.sigmoid(self.score_projection(self.dropout(hidden_states)))
        num_nodes = max(int(hidden_states.shape[1] * self.ratio), 1)
        indices = scores.topk(num_nodes, dim=1).indices.expand(
            -1,
            -1,
            hidden_states.shape[-1],
        )
        return torch.gather(hidden_states * scores, 1, indices)
