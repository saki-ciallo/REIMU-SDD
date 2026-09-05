from __future__ import annotations

from transformers import PreTrainedModel

from ...configuration.components.frontends import (
    LinearFrontendConfig,
    SincNetFrontendConfig,
    SSLFrontendConfig,
)
from .linear import LinearFrontendModel
from .sincnet import SincNetFrontendModel
from .ssl import SSLFrontendModel


def build_frontend(
    config: LinearFrontendConfig | SincNetFrontendConfig | SSLFrontendConfig,
) -> PreTrainedModel:
    match config:
        case LinearFrontendConfig():
            return LinearFrontendModel(config)
        case SincNetFrontendConfig():
            return SincNetFrontendModel(config)
        case SSLFrontendConfig():
            return SSLFrontendModel(config)
        case _:
            raise TypeError(f"Unsupported frontend config: {type(config).__name__}.")
