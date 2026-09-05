from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any

from transformers import PretrainedConfig

from .components.aasist import AASISTConfig
from .components.add import ADDConfig
from .components.backbone import AttentionBackboneConfig
from .components.classifier import LinearClassifierConfig
from .components.frontends import LinearFrontendConfig, SincNetFrontendConfig, SSLFrontendConfig
from .components.mixers import (
    AttentionFLAConfig,
    GatedDelta2FLAConfig,
    Mamba3FLAConfig,
    RavenFLAConfig,
)
from .components.pooling import GatedAttentionPoolingConfig

type ConfigValues = Mapping[str, Any]

FRONTEND_CONFIG_TYPES: dict[str, type[PretrainedConfig]] = {
    config_type.model_type: config_type
    for config_type in (
        SincNetFrontendConfig,
        LinearFrontendConfig,
        SSLFrontendConfig,
    )
}


def _mapping(value: object, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping, got {type(value).__name__}.")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{field_name} keys must be strings.")
    return dict(value)


def _constructor_fields(config_type: type[PretrainedConfig]) -> set[str]:
    return {
        name
        for name, parameter in inspect.signature(config_type.__init__).parameters.items()
        if name != "self"
        and parameter.kind not in {inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL}
    }


def _validate_fields(
    config_type: type[PretrainedConfig],
    values: Mapping[str, Any],
    *,
    field_name: str,
) -> None:
    unknown = set(values) - _constructor_fields(config_type) - {"model_type"}
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"{field_name} contains unknown fields: {names}.")


def _validated_nested(
    value: object,
    config_type: type[PretrainedConfig],
    *,
    field_name: str,
) -> dict[str, Any]:
    values = _mapping(value, field_name=field_name)
    expected_model_type = config_type.model_type
    model_type = values.get("model_type")
    compatible_types = {
        expected_model_type,
        *getattr(config_type, "legacy_model_types", ()),
    }
    if model_type not in compatible_types:
        supported = ", ".join(sorted(compatible_types))
        raise ValueError(f"{field_name}.model_type must be one of {supported}, got {model_type!r}.")
    _validate_fields(config_type, values, field_name=field_name)
    return values


def build_add_config(model_values: ConfigValues) -> ADDConfig:
    """Build one fully validated model config from the YAML ``model`` section."""

    values = _mapping(model_values, field_name="model")
    _validate_fields(ADDConfig, values, field_name="model")

    frontend_values = _mapping(
        values.get("frontend_config"),
        field_name="model.frontend_config",
    )
    frontend_model_type = frontend_values.get("model_type")
    frontend_type = FRONTEND_CONFIG_TYPES.get(frontend_model_type)
    if frontend_type is None:
        supported = ", ".join(FRONTEND_CONFIG_TYPES)
        raise ValueError(
            "model.frontend_config.model_type must be one of "
            f"{supported}, got {frontend_model_type!r}."
        )
    _validate_fields(
        frontend_type,
        frontend_values,
        field_name="model.frontend_config",
    )

    classifier_values = _validated_nested(
        values.get("classifier_config"),
        LinearClassifierConfig,
        field_name="model.classifier_config",
    )

    backbone_raw = values.get("backbone_config")
    if backbone_raw is not None:
        backbone_values = _validated_nested(
            backbone_raw,
            AttentionBackboneConfig,
            field_name="model.backbone_config",
        )
        for key, config_type in (
            ("attention_config", AttentionFLAConfig),
            ("raven_config", RavenFLAConfig),
            ("gdn2_config", GatedDelta2FLAConfig),
            ("mamba3_config", Mamba3FLAConfig),
        ):
            if key in backbone_values and backbone_values[key] is not None:
                _validated_nested(
                    backbone_values[key],
                    config_type,
                    field_name=f"model.backbone_config.{key}",
                )

    pooling_raw = values.get("pooling_config")
    if pooling_raw is not None:
        _validated_nested(
            pooling_raw,
            GatedAttentionPoolingConfig,
            field_name="model.pooling_config",
        )

    aasist_raw = values.get("aasist_config")
    if aasist_raw is not None:
        _validated_nested(
            aasist_raw,
            AASISTConfig,
            field_name="model.aasist_config",
        )

    values["frontend_config"] = frontend_values
    values["classifier_config"] = classifier_values
    return ADDConfig(**values)
