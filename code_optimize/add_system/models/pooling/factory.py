from __future__ import annotations

from torch import nn

from ...configuration.components.pooling import GatedAttentionPoolingConfig, PoolingType
from .gamp import GlobalAvgMaxPooling
from .gapv1 import GatedAttentionPoolingV1
from .gapv2 import GatedAttentionPoolingV2
from .mhgap import MultiHeadGatedAttentionPooling

POOLING_TYPES: dict[PoolingType, type[nn.Module]] = {
    PoolingType.MHGAP: MultiHeadGatedAttentionPooling,
    PoolingType.GAMP: GlobalAvgMaxPooling,
    PoolingType.GAPV1: GatedAttentionPoolingV1,
    PoolingType.GAPV2: GatedAttentionPoolingV2,
}


def build_pooling(config: GatedAttentionPoolingConfig) -> nn.Module:
    return POOLING_TYPES[PoolingType(config.pooling_type)](config)
