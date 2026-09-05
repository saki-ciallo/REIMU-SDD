from __future__ import annotations

from typing import ClassVar

import torch
from torch import nn
from transformers import PreTrainedModel

from ...configuration.components.aasist import AASISTConfig
from ..outputs import AASISTOutput
from .encoder import AASISTEncoder
from .graph import GraphAttentionLayer, HeterogeneousGraphAttentionLayer


class AASISTModel(PreTrainedModel):
    config_class = AASISTConfig
    base_model_prefix = "aasist"
    main_input_name = "input_ids"
    _no_split_modules: ClassVar[list[str]] = [
        "AASISTEncoder",
        "GraphAttentionLayer",
        "HeterogeneousGraphAttentionLayer",
        "ResidualConvBlock",
    ]

    def __init__(self, config: AASISTConfig) -> None:
        super().__init__(config)
        self.encoder = AASISTEncoder(config)
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, GraphAttentionLayer):
            nn.init.xavier_normal_(module.attention_weight)
        elif isinstance(module, HeterogeneousGraphAttentionLayer):
            nn.init.xavier_normal_(module.attention_weight11)
            nn.init.xavier_normal_(module.attention_weight22)
            nn.init.xavier_normal_(module.attention_weight12)
            nn.init.xavier_normal_(module.master_attention_weight)
        elif isinstance(module, AASISTEncoder):
            nn.init.normal_(module.spectral_position)
            nn.init.normal_(module.master1)
            nn.init.normal_(module.master2)

    def forward(
        self,
        input_ids: torch.Tensor,
        return_dict: bool | None = None,
    ) -> tuple[torch.Tensor, ...] | AASISTOutput:
        use_return_dict = self.config.return_dict if return_dict is None else return_dict
        embedding, temporal, spectral, master = self.encoder(input_ids)
        if not use_return_dict:
            return embedding, temporal, spectral, master
        return AASISTOutput(
            last_hidden_state=embedding,
            temporal_hidden_state=temporal,
            spectral_hidden_state=spectral,
            master_hidden_state=master,
        )
