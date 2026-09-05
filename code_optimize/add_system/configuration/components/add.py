from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from transformers import PretrainedConfig

from ..blocks import PipelineType
from ..validation import coerce_config
from .aasist import AASISTConfig
from .backbone import AttentionBackboneConfig
from .classifier import LinearClassifierConfig
from .frontends import LinearFrontendConfig, SincNetFrontendConfig, SSLFrontendConfig
from .pooling import GatedAttentionPoolingConfig

type FrontendConfig = SincNetFrontendConfig | LinearFrontendConfig | SSLFrontendConfig
type ConfigInput = PretrainedConfig | Mapping[str, Any] | None


def _coerce_frontend(value: ConfigInput) -> FrontendConfig:
    if isinstance(value, (SincNetFrontendConfig, LinearFrontendConfig, SSLFrontendConfig)):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("frontend_config must be a supported frontend config or mapping.")
    model_type = value.get("model_type")
    config_types = (
        SincNetFrontendConfig,
        LinearFrontendConfig,
        SSLFrontendConfig,
    )
    for config_type in config_types:
        if model_type == config_type.model_type:
            return config_type(**{key: item for key, item in value.items() if key != "model_type"})
    supported = ", ".join(config_type.model_type for config_type in config_types)
    raise TypeError(f"frontend_config.model_type must be one of {supported}, got {model_type!r}.")


class ADDConfig(PretrainedConfig):
    """Top-level Hugging Face config with explicit pipeline contracts."""

    model_type = "add"
    has_no_defaults_at_init = True
    keys_to_ignore_at_inference: ClassVar[list[str]] = [
        "loss_logits",
        "pooled_output",
        "past_key_values",
        "hidden_states",
        "attentions",
        "aux_loss",
        "temporal_hidden_state",
        "spectral_hidden_state",
        "master_hidden_state",
    ]

    def __init__(
        self,
        frontend_config: FrontendConfig | Mapping[str, Any],
        classifier_config: LinearClassifierConfig | Mapping[str, Any],
        backbone_config: AttentionBackboneConfig | Mapping[str, Any] | None = None,
        pooling_config: GatedAttentionPoolingConfig | Mapping[str, Any] | None = None,
        aasist_config: AASISTConfig | Mapping[str, Any] | None = None,
        model_architecture: str = "backbone_pooling",
        architecture_type: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        try:
            pipeline = PipelineType(model_architecture.strip().lower())
        except (AttributeError, ValueError) as exc:
            supported = ", ".join(member.value for member in PipelineType)
            raise ValueError(
                f"model_architecture must be one of {supported}, got {model_architecture!r}."
            ) from exc

        frontend = _coerce_frontend(frontend_config)
        classifier = coerce_config(
            classifier_config,
            LinearClassifierConfig,
            field_name="classifier_config",
        )
        backbone = coerce_config(
            backbone_config,
            AttentionBackboneConfig,
            field_name="backbone_config",
        )
        pooling = coerce_config(
            pooling_config,
            GatedAttentionPoolingConfig,
            field_name="pooling_config",
        )
        aasist = coerce_config(
            aasist_config,
            AASISTConfig,
            field_name="aasist_config",
        )

        if pipeline is PipelineType.BACKBONE_POOLING:
            if backbone is None or pooling is None:
                raise TypeError("backbone_pooling requires backbone_config and pooling_config.")
            if aasist is not None:
                raise ValueError("aasist_config must be omitted for backbone_pooling.")
            if frontend.frontend_output_dim != backbone.hidden_size:
                raise ValueError(
                    "frontend output dimension must equal backbone hidden_size; "
                    f"got {frontend.frontend_output_dim} and {backbone.hidden_size}."
                )
            if backbone.hidden_size != pooling.input_size:
                raise ValueError(
                    "backbone hidden_size must equal pooling input_size; "
                    f"got {backbone.hidden_size} and {pooling.input_size}."
                )
            if pooling.output_size != classifier.input_size:
                raise ValueError(
                    "pooling output_size must equal classifier input_size; "
                    f"got {pooling.output_size} and {classifier.input_size}."
                )
            if architecture_type is not None and architecture_type != backbone.architecture_type:
                raise ValueError("architecture_type must match backbone_config.architecture_type.")
            resolved_architecture = backbone.architecture_type
        else:
            if not isinstance(frontend, SSLFrontendConfig):
                raise TypeError("ssl_aasist requires SSLFrontendConfig.")
            if aasist is None:
                raise TypeError("ssl_aasist requires aasist_config.")
            if backbone is not None or pooling is not None:
                raise ValueError(
                    "backbone_config and pooling_config must be omitted for ssl_aasist."
                )
            if frontend.frontend_output_dim != aasist.input_size:
                raise ValueError("SSL frontend output dimension must equal AASIST input_size.")
            if aasist.output_size != classifier.input_size:
                raise ValueError("AASIST output_size must equal classifier input_size.")
            if architecture_type not in {None, PipelineType.SSL_AASIST.value}:
                raise ValueError("architecture_type must be 'ssl_aasist' for ssl_aasist.")
            resolved_architecture = PipelineType.SSL_AASIST.value

        self.model_architecture = pipeline.value
        self.architecture_type = resolved_architecture
        self.frontend_config = frontend
        self.backbone_config = backbone
        self.pooling_config = pooling
        self.aasist_config = aasist
        self.classifier_config = classifier
