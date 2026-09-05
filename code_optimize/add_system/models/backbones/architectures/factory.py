from __future__ import annotations

from ....configuration.blocks import ArchitectureType
from ....configuration.components.backbone import AttentionBackboneConfig
from .base import BackboneArchitecture
from .baseline import BaselineArchitecture
from .heterogeneous_hrm import HeterogeneousHRMArchitecture
from .hrm import HRMArchitecture
from .looped import LoopedArchitecture


def build_architecture(config: AttentionBackboneConfig) -> BackboneArchitecture:
    match ArchitectureType(config.architecture_type):
        case ArchitectureType.BASELINE:
            return BaselineArchitecture(config)
        case ArchitectureType.LOOPED:
            return LoopedArchitecture(config)
        case ArchitectureType.HRM:
            return HRMArchitecture(config)
        case ArchitectureType.HETEROGENEOUS_HRM:
            return HeterogeneousHRMArchitecture(config)
