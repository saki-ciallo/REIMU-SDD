from .factory import build_pooling
from .gamp import GlobalAvgMaxPooling
from .gapv1 import GatedAttentionPoolingV1
from .gapv2 import GatedAttentionPoolingV2
from .mhgap import MultiHeadGatedAttentionPooling
from .pooling import PoolingModel

__all__ = [
    "GatedAttentionPoolingV1",
    "GatedAttentionPoolingV2",
    "GlobalAvgMaxPooling",
    "MultiHeadGatedAttentionPooling",
    "PoolingModel",
    "build_pooling",
]
