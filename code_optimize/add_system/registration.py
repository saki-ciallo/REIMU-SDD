"""Explicit Hugging Face AutoClass registration for the optimized ADD model."""

from transformers import AutoConfig, AutoModel

from .configuration.components.add import ADDConfig
from .models import ADDModel


def register_auto_classes() -> None:
    """Register ADDConfig and ADDModel without import-time global side effects."""

    AutoConfig.register(ADDConfig.model_type, ADDConfig, exist_ok=True)
    AutoModel.register(ADDConfig, ADDModel, exist_ok=True)
