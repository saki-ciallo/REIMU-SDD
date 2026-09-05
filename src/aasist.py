from __future__ import annotations

import torch
from torch import nn

from .configuration import AASISTConfig


class GraphAttentionLayer(nn.Module):
    """Homogeneous graph attention over one node type."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        temperature: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = nn.Parameter(torch.empty(out_dim, 1))
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.batch_norm = nn.BatchNorm1d(out_dim)
        self.input_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.activation = nn.SELU(inplace=True)
        self.temperature = temperature

    @staticmethod
    def _pairwise_multiply_nodes(hidden_states: torch.Tensor) -> torch.Tensor:
        num_nodes = hidden_states.size(1)
        left = hidden_states.unsqueeze(2).expand(-1, -1, num_nodes, -1)
        return left * left.transpose(1, 2)

    def _derive_attention_map(self, hidden_states: torch.Tensor) -> torch.Tensor:
        attention_map = self._pairwise_multiply_nodes(hidden_states)
        attention_map = torch.tanh(self.att_proj(attention_map))
        attention_map = torch.matmul(attention_map, self.att_weight)
        return torch.softmax(
            (attention_map / self.temperature).float(),
            dim=-2,
        ).to(hidden_states.dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.input_dropout(hidden_states)
        attention_map = self._derive_attention_map(hidden_states)
        attended = torch.matmul(attention_map.squeeze(-1), hidden_states)
        hidden_states = self.proj_with_att(attended) + self.proj_without_att(hidden_states)
        original_shape = hidden_states.shape
        hidden_states = self.batch_norm(hidden_states.reshape(-1, original_shape[-1]))
        hidden_states = hidden_states.reshape(original_shape)
        return self.activation(hidden_states)


class HeterogeneousGraphAttentionLayer(nn.Module):
    """Graph attention across temporal, spectral, and master node types."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        temperature: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.proj_type1 = nn.Linear(in_dim, in_dim)
        self.proj_type2 = nn.Linear(in_dim, in_dim)
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_proj_master = nn.Linear(in_dim, out_dim)
        self.att_weight11 = nn.Parameter(torch.empty(out_dim, 1))
        self.att_weight22 = nn.Parameter(torch.empty(out_dim, 1))
        self.att_weight12 = nn.Parameter(torch.empty(out_dim, 1))
        self.att_weight_master = nn.Parameter(torch.empty(out_dim, 1))
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.proj_with_att_master = nn.Linear(in_dim, out_dim)
        self.proj_without_att_master = nn.Linear(in_dim, out_dim)
        self.batch_norm = nn.BatchNorm1d(out_dim)
        self.input_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.activation = nn.SELU(inplace=True)
        self.temperature = temperature

    @staticmethod
    def _pairwise_multiply_nodes(hidden_states: torch.Tensor) -> torch.Tensor:
        num_nodes = hidden_states.size(1)
        left = hidden_states.unsqueeze(2).expand(-1, -1, num_nodes, -1)
        return left * left.transpose(1, 2)

    def _derive_attention_map(
        self,
        hidden_states: torch.Tensor,
        num_type1: int,
        num_type2: int,
    ) -> torch.Tensor:
        attention_features = torch.tanh(
            self.att_proj(self._pairwise_multiply_nodes(hidden_states))
        )
        type11 = torch.matmul(
            attention_features[:, :num_type1, :num_type1],
            self.att_weight11,
        )
        type12 = torch.matmul(
            attention_features[:, :num_type1, num_type1:],
            self.att_weight12,
        )
        type21 = torch.matmul(
            attention_features[:, num_type1:, :num_type1],
            self.att_weight12,
        )
        type22 = torch.matmul(
            attention_features[:, num_type1:, num_type1:],
            self.att_weight22,
        )
        top = torch.cat((type11, type12), dim=2)
        bottom = torch.cat((type21, type22), dim=2)
        attention_map = torch.cat((top, bottom), dim=1)
        return torch.softmax(
            (attention_map / self.temperature).float(),
            dim=-2,
        ).to(hidden_states.dtype)

    def _update_master(
        self,
        hidden_states: torch.Tensor,
        master: torch.Tensor,
    ) -> torch.Tensor:
        attention_map = torch.tanh(self.att_proj_master(hidden_states * master))
        attention_map = torch.matmul(attention_map, self.att_weight_master)
        attention_map = torch.softmax(
            (attention_map / self.temperature).float(),
            dim=-2,
        ).to(hidden_states.dtype)
        attended = torch.matmul(
            attention_map.squeeze(-1).unsqueeze(1),
            hidden_states,
        )
        return self.proj_with_att_master(attended) + self.proj_without_att_master(master)

    def forward(
        self,
        type1_states: torch.Tensor,
        type2_states: torch.Tensor,
        master: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_type1 = type1_states.size(1)
        num_type2 = type2_states.size(1)
        type1_states = self.proj_type1(type1_states)
        type2_states = self.proj_type2(type2_states)
        hidden_states = torch.cat((type1_states, type2_states), dim=1)
        hidden_states = self.input_dropout(hidden_states)
        attention_map = self._derive_attention_map(
            hidden_states,
            num_type1=num_type1,
            num_type2=num_type2,
        )
        master = self._update_master(hidden_states, master)
        attended = torch.matmul(attention_map.squeeze(-1), hidden_states)
        hidden_states = self.proj_with_att(attended) + self.proj_without_att(hidden_states)
        original_shape = hidden_states.shape
        hidden_states = self.batch_norm(hidden_states.reshape(-1, original_shape[-1]))
        hidden_states = self.activation(hidden_states.reshape(original_shape))
        return (
            hidden_states.narrow(1, 0, num_type1),
            hidden_states.narrow(1, num_type1, num_type2),
            master,
        )


class GraphPool(nn.Module):
    """Select the highest-scoring fraction of graph nodes."""

    def __init__(self, ratio: float, input_size: int, dropout: float) -> None:
        super().__init__()
        self.ratio = ratio
        self.score_proj = nn.Linear(input_size, 1)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        scores = torch.sigmoid(self.score_proj(self.dropout(hidden_states)))
        num_nodes = max(int(hidden_states.size(1) * self.ratio), 1)
        indices = torch.topk(scores, num_nodes, dim=1).indices
        indices = indices.expand(-1, -1, hidden_states.size(-1))
        return torch.gather(hidden_states * scores, 1, indices)


class ResidualBlock(nn.Module):
    """Residual 2D convolution block used by the AASIST encoder stem."""

    def __init__(self, channels: list[int], first: bool = False) -> None:
        super().__init__()
        self.first = first
        if not first:
            self.batch_norm1 = nn.BatchNorm2d(channels[0])
        self.conv1 = nn.Conv2d(
            channels[0],
            channels[1],
            kernel_size=(2, 3),
            padding=(1, 1),
        )
        self.activation = nn.SELU(inplace=True)
        self.batch_norm2 = nn.BatchNorm2d(channels[1])
        self.conv2 = nn.Conv2d(
            channels[1],
            channels[1],
            kernel_size=(2, 3),
            padding=(0, 1),
        )
        self.downsample = (
            nn.Conv2d(
                channels[0],
                channels[1],
                kernel_size=(1, 3),
                padding=(0, 1),
            )
            if channels[0] != channels[1]
            else nn.Identity()
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = self.downsample(hidden_states)
        if not self.first:
            hidden_states = self.activation(self.batch_norm1(hidden_states))
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.conv2(self.activation(self.batch_norm2(hidden_states)))
        return hidden_states + residual


class AASISTEncoder(nn.Module):
    """Convert frame-level SSL features into one AASIST utterance embedding."""

    def __init__(self, config: AASISTConfig) -> None:
        super().__init__()
        feature_size = config.feature_size
        encoder_output_size = config.encoder_output_size
        gat_input_size, gat_output_size = config.gat_dims
        channel_pairs = config.filts[1:]

        self.config = config
        self.input_projection = nn.Linear(config.input_size, feature_size)
        self.initial_batch_norm = nn.BatchNorm2d(1)
        self.initial_pool = nn.MaxPool2d(kernel_size=(3, 3))
        self.activation = nn.SELU(inplace=True)
        self.encoder = nn.Sequential(
            ResidualBlock(channel_pairs[0], first=True),
            ResidualBlock(channel_pairs[1]),
            ResidualBlock(channel_pairs[2]),
            ResidualBlock(channel_pairs[3]),
            ResidualBlock(channel_pairs[3]),
            ResidualBlock(channel_pairs[3]),
        )
        self.encoder_batch_norm = nn.BatchNorm2d(encoder_output_size)
        self.attention = nn.Sequential(
            nn.Conv2d(encoder_output_size, feature_size, kernel_size=1),
            nn.SELU(inplace=True),
            nn.BatchNorm2d(feature_size),
            nn.Conv2d(feature_size, encoder_output_size, kernel_size=1),
        )

        self.spectral_position = nn.Parameter(
            torch.empty(1, config.spectral_num_nodes, encoder_output_size)
        )
        self.master1 = nn.Parameter(torch.empty(1, 1, gat_input_size))
        self.master2 = nn.Parameter(torch.empty(1, 1, gat_input_size))

        self.spectral_gat = GraphAttentionLayer(
            encoder_output_size,
            gat_input_size,
            temperature=config.temperatures[0],
            dropout=config.graph_attention_dropout,
        )
        self.temporal_gat = GraphAttentionLayer(
            encoder_output_size,
            gat_input_size,
            temperature=config.temperatures[1],
            dropout=config.graph_attention_dropout,
        )
        self.spectral_pool = GraphPool(
            config.pool_ratios[0],
            gat_input_size,
            config.graph_pool_dropout,
        )
        self.temporal_pool = GraphPool(
            config.pool_ratios[1],
            gat_input_size,
            config.graph_pool_dropout,
        )

        self.heterogeneous_gat11 = HeterogeneousGraphAttentionLayer(
            gat_input_size,
            gat_output_size,
            temperature=config.temperatures[2],
            dropout=config.graph_attention_dropout,
        )
        self.heterogeneous_gat12 = HeterogeneousGraphAttentionLayer(
            gat_output_size,
            gat_output_size,
            temperature=config.temperatures[2],
            dropout=config.graph_attention_dropout,
        )
        self.heterogeneous_gat21 = HeterogeneousGraphAttentionLayer(
            gat_input_size,
            gat_output_size,
            temperature=config.temperatures[3],
            dropout=config.graph_attention_dropout,
        )
        self.heterogeneous_gat22 = HeterogeneousGraphAttentionLayer(
            gat_output_size,
            gat_output_size,
            temperature=config.temperatures[3],
            dropout=config.graph_attention_dropout,
        )
        self.spectral_pool1 = GraphPool(
            config.pool_ratios[2],
            gat_output_size,
            config.graph_pool_dropout,
        )
        self.temporal_pool1 = GraphPool(
            config.pool_ratios[2],
            gat_output_size,
            config.graph_pool_dropout,
        )
        self.spectral_pool2 = GraphPool(
            config.pool_ratios[3],
            gat_output_size,
            config.graph_pool_dropout,
        )
        self.temporal_pool2 = GraphPool(
            config.pool_ratios[3],
            gat_output_size,
            config.graph_pool_dropout,
        )
        self.path_dropout = (
            nn.Dropout(config.path_dropout) if config.path_dropout > 0.0 else nn.Identity()
        )
        self.output_dropout = (
            nn.Dropout(config.output_dropout) if config.output_dropout > 0.0 else nn.Identity()
        )

    def _run_graph_path(
        self,
        temporal_states: torch.Tensor,
        spectral_states: torch.Tensor,
        master: torch.Tensor,
        first_gat: HeterogeneousGraphAttentionLayer,
        second_gat: HeterogeneousGraphAttentionLayer,
        temporal_pool: GraphPool,
        spectral_pool: GraphPool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        temporal_states, spectral_states, master = first_gat(
            temporal_states,
            spectral_states,
            master,
        )
        temporal_states = temporal_pool(temporal_states)
        spectral_states = spectral_pool(spectral_states)
        temporal_update, spectral_update, master_update = second_gat(
            temporal_states,
            spectral_states,
            master,
        )
        return (
            temporal_states + temporal_update,
            spectral_states + spectral_update,
            master + master_update,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if hidden_states.ndim != 3:
            raise ValueError(
                "hidden_states must have shape [batch_size, seq_len, input_size]."
            )
        if hidden_states.shape[-1] != self.config.input_size:
            raise ValueError(
                f"hidden_states last dimension must match input_size={self.config.input_size}, "
                f"got {hidden_states.shape[-1]}."
            )

        hidden_states = self.input_projection(hidden_states)
        hidden_states = hidden_states.transpose(1, 2).unsqueeze(1)
        if hidden_states.shape[-1] < 3:
            raise ValueError("AASIST requires at least three input frames.")
        hidden_states = self.initial_pool(hidden_states)
        hidden_states = self.activation(self.initial_batch_norm(hidden_states))
        hidden_states = self.encoder(hidden_states)
        hidden_states = self.activation(self.encoder_batch_norm(hidden_states))
        attention = self.attention(hidden_states)

        spectral_weights = torch.softmax(attention.float(), dim=-1).to(hidden_states.dtype)
        spectral_states = torch.sum(hidden_states * spectral_weights, dim=-1).transpose(1, 2)
        if spectral_states.size(1) != self.spectral_position.size(1):
            raise RuntimeError(
                "AASIST spectral node count does not match its position embedding: "
                f"{spectral_states.size(1)} != {self.spectral_position.size(1)}."
            )
        spectral_states = spectral_states + self.spectral_position
        spectral_states = self.spectral_pool(self.spectral_gat(spectral_states))

        temporal_weights = torch.softmax(attention.float(), dim=-2).to(hidden_states.dtype)
        temporal_states = torch.sum(hidden_states * temporal_weights, dim=-2).transpose(1, 2)
        temporal_states = self.temporal_pool(self.temporal_gat(temporal_states))

        batch_size = hidden_states.size(0)
        temporal1, spectral1, master1 = self._run_graph_path(
            temporal_states,
            spectral_states,
            self.master1.expand(batch_size, -1, -1),
            self.heterogeneous_gat11,
            self.heterogeneous_gat12,
            self.temporal_pool1,
            self.spectral_pool1,
        )
        temporal2, spectral2, master2 = self._run_graph_path(
            temporal_states,
            spectral_states,
            self.master2.expand(batch_size, -1, -1),
            self.heterogeneous_gat21,
            self.heterogeneous_gat22,
            self.temporal_pool2,
            self.spectral_pool2,
        )

        temporal_states = torch.maximum(
            self.path_dropout(temporal1),
            self.path_dropout(temporal2),
        )
        spectral_states = torch.maximum(
            self.path_dropout(spectral1),
            self.path_dropout(spectral2),
        )
        master = torch.maximum(
            self.path_dropout(master1),
            self.path_dropout(master2),
        )

        embedding = torch.cat(
            (
                temporal_states.abs().amax(dim=1),
                temporal_states.mean(dim=1),
                spectral_states.abs().amax(dim=1),
                spectral_states.mean(dim=1),
                master.squeeze(1),
            ),
            dim=-1,
        )
        return (
            self.output_dropout(embedding),
            temporal_states,
            spectral_states,
            master,
        )


__all__ = [
    "AASISTEncoder",
    "GraphAttentionLayer",
    "GraphPool",
    "HeterogeneousGraphAttentionLayer",
    "ResidualBlock",
]
