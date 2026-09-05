"""Public Python configuration API.

The optimized ``code_optimize/configs/`` directory stores YAML experiment profiles.
This package contains their typed component configs, validation, and resolution.
"""

from .blocks import (
    ArchitectureType,
    BlockSpec,
    FeedForwardType,
    MixerType,
    PipelineType,
    expand_block_specs,
    normalize_block_specs,
)
from .components import (
    AASISTConfig,
    ADDConfig,
    AttentionBackboneConfig,
    AttentionFLAConfig,
    ClassifierType,
    GatedAttentionPoolingConfig,
    GatedDelta2FLAConfig,
    LinearClassifierConfig,
    LinearFrontendConfig,
    Mamba3FLAConfig,
    PoolingType,
    RavenFLAConfig,
    SincNetFrontendConfig,
    SSLFrontendConfig,
)
from .experiments import ResolvedExperiment, resolve_experiment
from .factory import build_add_config
from .settings import (
    AugmentationSettings,
    DataSettings,
    ExperimentSettings,
    LossSettings,
    RunSettings,
    TrainingSettings,
    build_experiment_settings,
)

__all__ = [
    "AASISTConfig",
    "ADDConfig",
    "ArchitectureType",
    "AttentionBackboneConfig",
    "AttentionFLAConfig",
    "AugmentationSettings",
    "BlockSpec",
    "ClassifierType",
    "DataSettings",
    "ExperimentSettings",
    "FeedForwardType",
    "GatedAttentionPoolingConfig",
    "GatedDelta2FLAConfig",
    "LinearClassifierConfig",
    "LinearFrontendConfig",
    "LossSettings",
    "Mamba3FLAConfig",
    "MixerType",
    "PipelineType",
    "PoolingType",
    "RavenFLAConfig",
    "ResolvedExperiment",
    "RunSettings",
    "SSLFrontendConfig",
    "SincNetFrontendConfig",
    "TrainingSettings",
    "build_add_config",
    "build_experiment_settings",
    "expand_block_specs",
    "normalize_block_specs",
    "resolve_experiment",
]
