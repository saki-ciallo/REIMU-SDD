from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from .aasist import (
    AASISTEncoder,
    GraphAttentionLayer,
    HeterogeneousGraphAttentionLayer,
)
from .configuration import AASISTConfig


@dataclass
class AASISTOutput(ModelOutput):
    last_hidden_state: Optional[torch.Tensor] = None
    temporal_hidden_state: Optional[torch.Tensor] = None
    spectral_hidden_state: Optional[torch.Tensor] = None
    master_hidden_state: Optional[torch.Tensor] = None


class AASISTSummaryMixin:
    def parameter_name_summary(self) -> List[Dict[str, object]]:
        return [
            {
                "name": name,
                "shape": tuple(parameter.shape),
                "dtype": str(parameter.dtype).replace("torch.", ""),
                "requires_grad": parameter.requires_grad,
            }
            for name, parameter in self.named_parameters()
        ]

    def architecture_summary(self) -> str:
        return "\n".join(
            [
                "AASISTModel",
                f"  input_size: {self.config.input_size}",
                f"  filts: {self.config.filts}",
                f"  gat_dims: {self.config.gat_dims}",
                f"  pool_ratios: {self.config.pool_ratios}",
                f"  temperatures: {self.config.temperatures}",
                f"  spectral_num_nodes: {self.config.spectral_num_nodes}",
                f"  output_size: {self.config.output_size}",
                "  readout: concat(T_max, T_avg, S_max, S_avg, master)",
            ]
        )

    def num_trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


class AASISTModel(AASISTSummaryMixin, PreTrainedModel):
    config_class = AASISTConfig
    base_model_prefix = "aasist"
    main_input_name = "input_ids"
    _no_split_modules = [
        "AASISTEncoder",
        "GraphAttentionLayer",
        "HeterogeneousGraphAttentionLayer",
        "ResidualBlock",
    ]

    def __init__(self, config: AASISTConfig) -> None:
        super().__init__(config)
        self.aasist = AASISTEncoder(config)
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, GraphAttentionLayer):
            nn.init.xavier_normal_(module.att_weight)
        elif isinstance(module, HeterogeneousGraphAttentionLayer):
            nn.init.xavier_normal_(module.att_weight11)
            nn.init.xavier_normal_(module.att_weight22)
            nn.init.xavier_normal_(module.att_weight12)
            nn.init.xavier_normal_(module.att_weight_master)
        elif isinstance(module, AASISTEncoder):
            nn.init.normal_(module.spectral_position)
            nn.init.normal_(module.master1)
            nn.init.normal_(module.master2)

    def forward(
        self,
        input_ids: torch.Tensor,
        return_dict: Optional[bool] = None,
    ) -> tuple | AASISTOutput:
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        (
            embedding,
            temporal_hidden_state,
            spectral_hidden_state,
            master_hidden_state,
        ) = self.aasist(input_ids)

        if not return_dict:
            return (
                embedding,
                temporal_hidden_state,
                spectral_hidden_state,
                master_hidden_state,
            )
        return AASISTOutput(
            last_hidden_state=embedding,
            temporal_hidden_state=temporal_hidden_state,
            spectral_hidden_state=spectral_hidden_state,
            master_hidden_state=master_hidden_state,
        )


__all__ = [
    "AASISTModel",
    "AASISTOutput",
    "AASISTSummaryMixin",
]
