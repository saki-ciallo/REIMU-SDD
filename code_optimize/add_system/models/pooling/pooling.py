from __future__ import annotations

from typing import ClassVar

import torch
from torch import nn
from transformers import PreTrainedModel

from ...configuration.components.pooling import GatedAttentionPoolingConfig
from ..outputs import PoolingOutput
from .base import PoolingOutputTransform
from .factory import POOLING_TYPES, build_pooling
from .mhgap import MultiHeadGatedAttentionPooling


class PoolingModel(PreTrainedModel):
    config_class = GatedAttentionPoolingConfig
    base_model_prefix = "pooling"
    _no_split_modules: ClassVar[list[str]] = [
        pooling_type.__name__ for pooling_type in POOLING_TYPES.values()
    ]

    def __init__(self, config: GatedAttentionPoolingConfig) -> None:
        super().__init__(config)
        self.pooling = build_pooling(config)
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.initializer_range,
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, MultiHeadGatedAttentionPooling):
            nn.init.normal_(
                module.score_weight,
                mean=0.0,
                std=self.config.initializer_range,
            )
        elif isinstance(module, PoolingOutputTransform):
            norm = module.norm
            if hasattr(norm, "weight"):
                nn.init.ones_(norm.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        return_dict: bool | None = None,
    ) -> tuple[torch.Tensor] | PoolingOutput:
        use_return_dict = self.config.return_dict if return_dict is None else return_dict
        pooled_output = self.pooling(input_ids)
        if not use_return_dict:
            return (pooled_output,)
        return PoolingOutput(pooled_output=pooled_output)
