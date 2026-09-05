from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from typing import Any

from transformers import PretrainedConfig

from ..validation import positive_float, positive_int, probability


class AASISTConfig(PretrainedConfig):
    """Configuration for the SSL-to-AASIST embedding backend."""

    model_type = "aasist"

    def __init__(
        self,
        input_size: int = 768,
        filts: Sequence[object] | None = None,
        gat_dims: Sequence[int] | None = None,
        pool_ratios: Sequence[float] | None = None,
        temperatures: Sequence[float] | None = None,
        graph_attention_dropout: float = 0.2,
        graph_pool_dropout: float = 0.3,
        path_dropout: float = 0.2,
        output_dropout: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.input_size = positive_int(input_size, field_name="input_size")
        self.filts = self._validate_filts(filts or (128, (1, 32), (32, 32), (32, 64), (64, 64)))
        self.gat_dims = self._fixed_positive_ints(
            gat_dims or (64, 32),
            field_name="gat_dims",
            length=2,
        )
        self.pool_ratios = self._fixed_ratios(pool_ratios or (0.5, 0.5, 0.5, 0.5))
        self.temperatures = self._fixed_positive_floats(
            temperatures or (2.0, 2.0, 100.0, 100.0),
            field_name="temperatures",
            length=4,
        )
        self.graph_attention_dropout = probability(
            graph_attention_dropout,
            field_name="graph_attention_dropout",
        )
        self.graph_pool_dropout = probability(
            graph_pool_dropout,
            field_name="graph_pool_dropout",
        )
        self.path_dropout = probability(path_dropout, field_name="path_dropout")
        self.output_dropout = probability(output_dropout, field_name="output_dropout")

    @staticmethod
    def _fixed_positive_ints(
        values: Sequence[int],
        *,
        field_name: str,
        length: int,
    ) -> list[int]:
        if len(values) != length:
            raise ValueError(f"{field_name} must contain {length} values.")
        return [
            positive_int(value, field_name=f"{field_name}[{index}]")
            for index, value in enumerate(values)
        ]

    @staticmethod
    def _fixed_positive_floats(
        values: Sequence[float],
        *,
        field_name: str,
        length: int,
    ) -> list[float]:
        if len(values) != length:
            raise ValueError(f"{field_name} must contain {length} values.")
        return [
            positive_float(value, field_name=f"{field_name}[{index}]")
            for index, value in enumerate(values)
        ]

    @classmethod
    def _validate_filts(cls, values: Sequence[object]) -> list[object]:
        if len(values) != 5:
            raise ValueError("filts must contain one feature size and four channel pairs.")
        feature_size = positive_int(values[0], field_name="filts[0]")
        if feature_size < 3:
            raise ValueError("filts[0] must be at least 3.")

        pairs: list[list[int]] = []
        for index, raw_pair in enumerate(values[1:], start=1):
            if (
                not isinstance(raw_pair, Sequence)
                or isinstance(raw_pair, (str, bytes))
                or len(raw_pair) != 2
            ):
                raise TypeError(f"filts[{index}] must be a two-element sequence.")
            pairs.append(
                cls._fixed_positive_ints(
                    raw_pair,
                    field_name=f"filts[{index}]",
                    length=2,
                )
            )
        if pairs[0][0] != 1:
            raise ValueError("filts[1][0] must be 1.")
        if any(previous[1] != current[0] for previous, current in pairwise(pairs)):
            raise ValueError("Adjacent filts channel pairs must be continuous.")
        if pairs[-1][0] != pairs[-1][1]:
            raise ValueError("The final filts channel pair must preserve its channel size.")
        return [feature_size, *pairs]

    @staticmethod
    def _fixed_ratios(values: Sequence[float]) -> list[float]:
        if len(values) != 4:
            raise ValueError("pool_ratios must contain four values.")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
            raise TypeError("pool_ratios must contain only numbers.")
        ratios = [float(value) for value in values]
        if any(not 0.0 < value <= 1.0 for value in ratios):
            raise ValueError("pool_ratios must lie in (0, 1].")
        return ratios

    @property
    def feature_size(self) -> int:
        return int(self.filts[0])

    @property
    def encoder_output_size(self) -> int:
        return int(self.filts[-1][-1])

    @property
    def spectral_num_nodes(self) -> int:
        return self.feature_size // 3

    @property
    def output_size(self) -> int:
        return 5 * self.gat_dims[1]
