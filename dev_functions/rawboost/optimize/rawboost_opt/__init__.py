"""Modern, independently testable RawBoost preprocessing backends."""

from .api import RawBoost
from .config import Algorithm, Backend, FIRBackend, RawBoostConfig
from .numpy_backend import augment_numpy
from .parameters import RawBoostParameters, sample_parameters
from .torch_backend import augment_torch

__all__ = [
    "Algorithm",
    "Backend",
    "FIRBackend",
    "RawBoost",
    "RawBoostConfig",
    "RawBoostParameters",
    "augment_numpy",
    "augment_torch",
    "sample_parameters",
]
