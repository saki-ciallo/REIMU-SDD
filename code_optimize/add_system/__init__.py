"""Independent modular implementation of the ADD training stack."""

from .configuration import (
    ADDConfig,
    ArchitectureType,
    AttentionBackboneConfig,
    BlockSpec,
    FeedForwardType,
    GatedAttentionPoolingConfig,
    LinearClassifierConfig,
    LinearFrontendConfig,
    MixerType,
    PipelineType,
    ResolvedExperiment,
    SincNetFrontendConfig,
    SSLFrontendConfig,
    resolve_experiment,
)

__all__ = [
    "ADDConfig",
    "ArchitectureType",
    "AttentionBackboneConfig",
    "BlockSpec",
    "FeedForwardType",
    "GatedAttentionPoolingConfig",
    "LinearClassifierConfig",
    "LinearFrontendConfig",
    "MixerType",
    "PipelineType",
    "ResolvedExperiment",
    "SSLFrontendConfig",
    "SincNetFrontendConfig",
    "resolve_experiment",
]
