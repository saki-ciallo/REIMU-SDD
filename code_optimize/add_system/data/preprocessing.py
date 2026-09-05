from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datasets import Dataset

    from ..configuration.settings import AugmentationSettings, DataSettings


def _build_original_rawboost(settings: AugmentationSettings):
    """Adapt structured settings to the bundled canonical RawBoost API."""

    try:
        from utilis.rawboost_utils import RawBoostArguments, RawBoostWaveformTransform
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The bundled CPU RawBoost package is unavailable. "
            "Run from code_optimize/train.sh or install code_optimize."
        ) from exc

    arguments = RawBoostArguments(
        rawboost_algo=settings.algorithm,
        rawboost_apply_to_validation=settings.apply_to_validation,
        rawboost_num_bands=settings.num_bands,
        rawboost_min_frequency=settings.min_frequency,
        rawboost_max_frequency=settings.max_frequency,
        rawboost_min_bandwidth=settings.min_bandwidth,
        rawboost_max_bandwidth=settings.max_bandwidth,
        rawboost_min_coefficients=settings.min_coefficients,
        rawboost_max_coefficients=settings.max_coefficients,
        rawboost_min_gain=settings.min_gain,
        rawboost_max_gain=settings.max_gain,
        rawboost_min_nonlinear_bias=settings.min_nonlinear_bias,
        rawboost_max_nonlinear_bias=settings.max_nonlinear_bias,
        rawboost_nonlinearity_order=settings.nonlinearity_order,
        rawboost_impulse_percent=settings.impulse_percent,
        rawboost_impulse_gain=settings.impulse_gain,
        rawboost_min_snr=settings.min_snr,
        rawboost_max_snr=settings.max_snr,
    )
    return RawBoostWaveformTransform(arguments, settings.sampling_rate)


def apply_rawboost_transforms(
    train_dataset: Dataset | None,
    eval_dataset: Dataset | None,
    *,
    augmentation: AugmentationSettings,
    data: DataSettings,
) -> tuple[Dataset | None, Dataset | None]:
    """Attach original CPU RawBoost transforms before collation and model input."""

    if not augmentation.enabled:
        return train_dataset, eval_dataset

    waveform_transform = _build_original_rawboost(augmentation)
    from utilis.rawboost_utils import build_rawboost_dataset_transform

    dataset_transform = build_rawboost_dataset_transform(
        waveform_transform,
        data.input_column,
    )
    if train_dataset is not None:
        train_dataset = train_dataset.with_transform(dataset_transform)
    if eval_dataset is not None and augmentation.apply_to_validation:
        eval_dataset = eval_dataset.with_transform(dataset_transform)
    return train_dataset, eval_dataset
