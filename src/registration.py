from __future__ import annotations

from transformers import AutoConfig, AutoModel

from .configuration import (
    ADDConfig,
    AASISTConfig,
    AttentionBackboneConfig,
    GatedAttentionPoolingConfig,
    LinearClassifierConfig,
    LinearFrontendConfig,
    SincNetFrontendConfig,
    SSLFrontendConfig,
)
from .modeling_aasist import AASISTModel
from .modeling_add import ADDModel
from .modeling_backbone import AttentionBackboneModel
from .modeling_classifier import LinearClassifierModel
from .modeling_linear import LinearFrontendModel
from .modeling_pooling import GatedAttentionPoolingModel
from .modeling_sincnet import SincNetFrontendModel
from .modeling_ssl import SSLFrontendModel


def register_audio_models_for_auto() -> None:
    """Register custom audio frontend and backbone models for AutoConfig/AutoModel."""
    AutoConfig.register(
        ADDConfig.model_type,
        ADDConfig,
        exist_ok=True,
    )
    AutoModel.register(ADDConfig, ADDModel, exist_ok=True)

    AutoConfig.register(
        SincNetFrontendConfig.model_type,
        SincNetFrontendConfig,
        exist_ok=True,
    )
    AutoModel.register(SincNetFrontendConfig, SincNetFrontendModel, exist_ok=True)

    AutoConfig.register(
        LinearFrontendConfig.model_type,
        LinearFrontendConfig,
        exist_ok=True,
    )
    AutoModel.register(LinearFrontendConfig, LinearFrontendModel, exist_ok=True)

    AutoConfig.register(
        SSLFrontendConfig.model_type,
        SSLFrontendConfig,
        exist_ok=True,
    )
    AutoModel.register(SSLFrontendConfig, SSLFrontendModel, exist_ok=True)

    AutoConfig.register(
        AASISTConfig.model_type,
        AASISTConfig,
        exist_ok=True,
    )
    AutoModel.register(AASISTConfig, AASISTModel, exist_ok=True)

    AutoConfig.register(
        AttentionBackboneConfig.model_type,
        AttentionBackboneConfig,
        exist_ok=True,
    )
    AutoModel.register(AttentionBackboneConfig, AttentionBackboneModel, exist_ok=True)

    AutoConfig.register(
        GatedAttentionPoolingConfig.model_type,
        GatedAttentionPoolingConfig,
        exist_ok=True,
    )
    AutoModel.register(
        GatedAttentionPoolingConfig,
        GatedAttentionPoolingModel,
        exist_ok=True,
    )

    AutoConfig.register(
        LinearClassifierConfig.model_type,
        LinearClassifierConfig,
        exist_ok=True,
    )
    AutoModel.register(
        LinearClassifierConfig,
        LinearClassifierModel,
        exist_ok=True,
    )


def register_sincnet_frontend_for_auto() -> None:
    """Frontend registration entrypoint."""

    AutoConfig.register(
        SincNetFrontendConfig.model_type,
        SincNetFrontendConfig,
        exist_ok=True,
    )
    AutoModel.register(SincNetFrontendConfig, SincNetFrontendModel, exist_ok=True)


def register_linear_frontend_for_auto() -> None:
    """Linear frontend registration entrypoint."""

    AutoConfig.register(
        LinearFrontendConfig.model_type,
        LinearFrontendConfig,
        exist_ok=True,
    )
    AutoModel.register(LinearFrontendConfig, LinearFrontendModel, exist_ok=True)


def register_ssl_frontend_for_auto() -> None:
    """SSL frontend registration entrypoint."""

    AutoConfig.register(
        SSLFrontendConfig.model_type,
        SSLFrontendConfig,
        exist_ok=True,
    )
    AutoModel.register(SSLFrontendConfig, SSLFrontendModel, exist_ok=True)


def register_aasist_for_auto() -> None:
    """AASIST backend registration entrypoint."""

    AutoConfig.register(
        AASISTConfig.model_type,
        AASISTConfig,
        exist_ok=True,
    )
    AutoModel.register(AASISTConfig, AASISTModel, exist_ok=True)


def register_pooling_for_auto() -> None:
    """Pooling registration entrypoint."""

    AutoConfig.register(
        GatedAttentionPoolingConfig.model_type,
        GatedAttentionPoolingConfig,
        exist_ok=True,
    )
    AutoModel.register(
        GatedAttentionPoolingConfig,
        GatedAttentionPoolingModel,
        exist_ok=True,
    )


def register_classifier_for_auto() -> None:
    """Classifier registration entrypoint."""

    AutoConfig.register(
        LinearClassifierConfig.model_type,
        LinearClassifierConfig,
        exist_ok=True,
    )
    AutoModel.register(
        LinearClassifierConfig,
        LinearClassifierModel,
        exist_ok=True,
    )


def register_add_for_auto() -> None:
    """Complete frontend + backbone registration entrypoint."""

    AutoConfig.register(
        ADDConfig.model_type,
        ADDConfig,
        exist_ok=True,
    )
    AutoModel.register(ADDConfig, ADDModel, exist_ok=True)
