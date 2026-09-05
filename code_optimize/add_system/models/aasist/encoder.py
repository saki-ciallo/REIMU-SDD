from __future__ import annotations

import torch
from torch import nn

from ...configuration.components.aasist import AASISTConfig
from .graph import GraphAttentionLayer, GraphPool, HeterogeneousGraphAttentionLayer


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: list[int], *, first: bool = False) -> None:
        super().__init__()
        self.input_norm = nn.Identity() if first else nn.BatchNorm2d(channels[0])
        self.conv1 = nn.Conv2d(
            channels[0],
            channels[1],
            kernel_size=(2, 3),
            padding=(1, 1),
        )
        self.activation = nn.SELU(inplace=True)
        self.output_norm = nn.BatchNorm2d(channels[1])
        self.conv2 = nn.Conv2d(
            channels[1],
            channels[1],
            kernel_size=(2, 3),
            padding=(0, 1),
        )
        self.residual_projection = (
            nn.Conv2d(channels[0], channels[1], (1, 3), padding=(0, 1))
            if channels[0] != channels[1]
            else nn.Identity()
        )
        self.first = first

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = self.residual_projection(hidden_states)
        if not self.first:
            hidden_states = self.activation(self.input_norm(hidden_states))
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.conv2(self.activation(self.output_norm(hidden_states)))
        return hidden_states + residual


class AASISTEncoder(nn.Module):
    def __init__(self, config: AASISTConfig) -> None:
        super().__init__()
        feature_size = config.feature_size
        encoder_size = config.encoder_output_size
        graph_input_size, graph_output_size = config.gat_dims
        pairs = config.filts[1:]
        self.config = config
        self.input_projection = nn.Linear(config.input_size, feature_size)
        self.initial_norm = nn.BatchNorm2d(1)
        self.initial_pool = nn.MaxPool2d((3, 3))
        self.activation = nn.SELU(inplace=True)
        self.convolution = nn.Sequential(
            ResidualConvBlock(pairs[0], first=True),
            ResidualConvBlock(pairs[1]),
            ResidualConvBlock(pairs[2]),
            ResidualConvBlock(pairs[3]),
            ResidualConvBlock(pairs[3]),
            ResidualConvBlock(pairs[3]),
        )
        self.encoder_norm = nn.BatchNorm2d(encoder_size)
        self.attention = nn.Sequential(
            nn.Conv2d(encoder_size, feature_size, 1),
            nn.SELU(inplace=True),
            nn.BatchNorm2d(feature_size),
            nn.Conv2d(feature_size, encoder_size, 1),
        )
        self.spectral_position = nn.Parameter(
            torch.empty(1, config.spectral_num_nodes, encoder_size)
        )
        self.master1 = nn.Parameter(torch.empty(1, 1, graph_input_size))
        self.master2 = nn.Parameter(torch.empty(1, 1, graph_input_size))
        self.spectral_attention = GraphAttentionLayer(
            encoder_size,
            graph_input_size,
            temperature=config.temperatures[0],
            dropout=config.graph_attention_dropout,
        )
        self.temporal_attention = GraphAttentionLayer(
            encoder_size,
            graph_input_size,
            temperature=config.temperatures[1],
            dropout=config.graph_attention_dropout,
        )
        self.spectral_pool = GraphPool(
            config.pool_ratios[0],
            graph_input_size,
            config.graph_pool_dropout,
        )
        self.temporal_pool = GraphPool(
            config.pool_ratios[1],
            graph_input_size,
            config.graph_pool_dropout,
        )
        self.path1 = self._make_path(config, graph_input_size, graph_output_size, 2)
        self.path2 = self._make_path(config, graph_input_size, graph_output_size, 3)
        self.path_dropout = nn.Dropout(config.path_dropout)
        self.output_dropout = nn.Dropout(config.output_dropout)

    @staticmethod
    def _make_path(
        config: AASISTConfig,
        input_size: int,
        output_size: int,
        temperature_index: int,
    ) -> nn.ModuleDict:
        return nn.ModuleDict(
            {
                "first": HeterogeneousGraphAttentionLayer(
                    input_size,
                    output_size,
                    temperature=config.temperatures[temperature_index],
                    dropout=config.graph_attention_dropout,
                ),
                "second": HeterogeneousGraphAttentionLayer(
                    output_size,
                    output_size,
                    temperature=config.temperatures[temperature_index],
                    dropout=config.graph_attention_dropout,
                ),
                "temporal_pool": GraphPool(
                    config.pool_ratios[temperature_index],
                    output_size,
                    config.graph_pool_dropout,
                ),
                "spectral_pool": GraphPool(
                    config.pool_ratios[temperature_index],
                    output_size,
                    config.graph_pool_dropout,
                ),
            }
        )

    @staticmethod
    def _run_path(
        path: nn.ModuleDict,
        temporal: torch.Tensor,
        spectral: torch.Tensor,
        master: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        temporal, spectral, master = path["first"](temporal, spectral, master)
        temporal = path["temporal_pool"](temporal)
        spectral = path["spectral_pool"](spectral)
        temporal_update, spectral_update, master_update = path["second"](
            temporal,
            spectral,
            master,
        )
        return (
            temporal + temporal_update,
            spectral + spectral_update,
            master + master_update,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.config.input_size:
            raise ValueError(
                f"hidden_states must have shape [batch, sequence, {self.config.input_size}]."
            )
        hidden_states = self.input_projection(hidden_states).transpose(1, 2)[:, None]
        if hidden_states.shape[-1] < 3:
            raise ValueError("AASIST requires at least three input frames.")
        hidden_states = self.initial_pool(hidden_states)
        hidden_states = self.activation(self.initial_norm(hidden_states))
        hidden_states = self.convolution(hidden_states)
        hidden_states = self.activation(self.encoder_norm(hidden_states))
        attention = self.attention(hidden_states)

        spectral_weights = torch.softmax(attention.float(), dim=-1).to(hidden_states.dtype)
        spectral = (hidden_states * spectral_weights).sum(dim=-1).transpose(1, 2)
        if spectral.shape[1] != self.spectral_position.shape[1]:
            raise RuntimeError("Spectral node count does not match AASIST position embedding.")
        spectral = self.spectral_pool(self.spectral_attention(spectral + self.spectral_position))
        temporal_weights = torch.softmax(attention.float(), dim=-2).to(hidden_states.dtype)
        temporal = (hidden_states * temporal_weights).sum(dim=-2).transpose(1, 2)
        temporal = self.temporal_pool(self.temporal_attention(temporal))

        batch_size = hidden_states.shape[0]
        temporal1, spectral1, master1 = self._run_path(
            self.path1,
            temporal,
            spectral,
            self.master1.expand(batch_size, -1, -1),
        )
        temporal2, spectral2, master2 = self._run_path(
            self.path2,
            temporal,
            spectral,
            self.master2.expand(batch_size, -1, -1),
        )
        temporal = torch.maximum(
            self.path_dropout(temporal1),
            self.path_dropout(temporal2),
        )
        spectral = torch.maximum(
            self.path_dropout(spectral1),
            self.path_dropout(spectral2),
        )
        master = torch.maximum(
            self.path_dropout(master1),
            self.path_dropout(master2),
        )
        embedding = torch.cat(
            (
                temporal.abs().amax(dim=1),
                temporal.mean(dim=1),
                spectral.abs().amax(dim=1),
                spectral.mean(dim=1),
                master.squeeze(1),
            ),
            dim=-1,
        )
        return self.output_dropout(embedding), temporal, spectral, master
