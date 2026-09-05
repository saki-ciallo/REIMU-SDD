from __future__ import annotations

from enum import StrEnum
from typing import Any

from transformers import PretrainedConfig

from ..validation import boolean, one_of, positive_float, positive_int


class ClassifierType(StrEnum):
    LINEAR = "linear"
    AMSOFTMAX = "amsoftmax"


class LinearClassifierConfig(PretrainedConfig):
    model_type = "linear_classifier"

    def __init__(
        self,
        input_size: int = 128,
        num_labels: int = 2,
        classifier_type: str = "linear",
        bias: bool = False,
        am_margin: float = 0.3,
        am_scale: float = 15.0,
        am_eps: float = 1e-6,
        initializer_range: float = 0.02,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.input_size = positive_int(input_size, field_name="input_size")
        self.num_labels = positive_int(num_labels, field_name="num_labels")
        self.classifier_type = ClassifierType(
            one_of(
                classifier_type,
                {member.value for member in ClassifierType},
                field_name="classifier_type",
            )
        ).value
        self.bias = boolean(bias, field_name="bias")
        if isinstance(am_margin, bool) or not isinstance(am_margin, (int, float)):
            raise TypeError("am_margin must be a number.")
        if am_margin < 0:
            raise ValueError("am_margin must be non-negative.")
        self.am_margin = float(am_margin)
        self.am_scale = positive_float(am_scale, field_name="am_scale")
        self.am_eps = positive_float(am_eps, field_name="am_eps")
        self.initializer_range = positive_float(
            initializer_range,
            field_name="initializer_range",
        )
