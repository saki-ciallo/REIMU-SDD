from __future__ import annotations

from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk

from ..configuration.settings import CODE_OPTIMIZE_ROOT, DataSettings


def _select_split(
    cached: DatasetDict,
    split_name: str,
    *,
    max_samples: int | None,
) -> Dataset:
    if split_name not in cached:
        raise KeyError(f"Dataset cache is missing requested split {split_name!r}.")
    dataset = cached[split_name]
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return dataset


def load_splits(
    settings: DataSettings,
    *,
    do_train: bool,
    do_eval: bool,
) -> tuple[Dataset | None, Dataset | None]:
    """Load the configured train and validation datasets from a disk cache."""

    cache_path = Path(settings.dataset_cache_path).expanduser()
    if not cache_path.is_absolute():
        # Config paths belong to the optimized package, independent of the
        # directory from which a caller imports or invokes the loader.
        cache_path = CODE_OPTIMIZE_ROOT / cache_path
    cached = load_from_disk(cache_path)
    if not isinstance(cached, DatasetDict):
        raise TypeError("dataset_cache_path must contain a DatasetDict.")

    train_dataset = (
        _select_split(
            cached,
            settings.train_split,
            max_samples=settings.max_train_samples,
        )
        if do_train
        else None
    )
    eval_dataset = (
        _select_split(
            cached,
            settings.validation_split,
            max_samples=settings.max_eval_samples,
        )
        if do_eval
        else None
    )
    return train_dataset, eval_dataset
