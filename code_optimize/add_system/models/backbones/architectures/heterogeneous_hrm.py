from __future__ import annotations

from ....configuration.components.backbone import AttentionBackboneConfig
from .hrm import HRMArchitecture


class HeterogeneousHRMArchitecture(HRMArchitecture):
    """HRM with independently specified high- and low-level block topologies."""

    def __init__(self, config: AttentionBackboneConfig) -> None:
        super().__init__(
            config,
            h_specs=config.hrm_h_block_specs,
            l_specs=config.hrm_l_block_specs,
        )
