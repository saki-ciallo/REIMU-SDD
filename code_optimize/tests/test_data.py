from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch
from datasets import Dataset, DatasetDict

from add_system.configuration import AugmentationSettings, DataSettings
from add_system.data import AudioClassificationCollator, apply_rawboost_transforms, load_splits

CODE_OPTIMIZE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_OPTIMIZE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_OPTIMIZE_ROOT))


def _settings(cache_path: str, **kwargs: object) -> DataSettings:
    return DataSettings(
        dataset_key="test",
        dataset_cache_path=cache_path,
        **kwargs,
    )


def _dataset(length: int) -> Dataset:
    return Dataset.from_dict(
        {
            "input_values": [[float(index), float(index + 1)] for index in range(length)],
            "labels": [index % 2 for index in range(length)],
        }
    )


class DatasetLoadingTests(unittest.TestCase):
    def test_load_splits_selects_configured_splits_and_limits(self) -> None:
        with TemporaryDirectory() as directory:
            cache_path = Path(directory) / "cache"
            DatasetDict(train=_dataset(4), validation=_dataset(3)).save_to_disk(cache_path)

            train, evaluation = load_splits(
                _settings(
                    str(cache_path),
                    max_train_samples=2,
                    max_eval_samples=1,
                ),
                do_train=True,
                do_eval=True,
            )

            self.assertIsNotNone(train)
            self.assertIsNotNone(evaluation)
            self.assertEqual(len(train), 2)
            self.assertEqual(len(evaluation), 1)

    def test_load_splits_skips_disabled_split(self) -> None:
        with TemporaryDirectory() as directory:
            cache_path = Path(directory) / "cache"
            DatasetDict(train=_dataset(2)).save_to_disk(cache_path)

            train, evaluation = load_splits(
                _settings(str(cache_path)),
                do_train=True,
                do_eval=False,
            )

            self.assertIsNotNone(train)
            self.assertIsNone(evaluation)

    def test_load_splits_rejects_non_dataset_dict_cache(self) -> None:
        with TemporaryDirectory() as directory:
            cache_path = Path(directory) / "cache"
            _dataset(1).save_to_disk(cache_path)

            with self.assertRaisesRegex(TypeError, "DatasetDict"):
                load_splits(_settings(str(cache_path)), do_train=True, do_eval=False)

    def test_load_splits_reports_missing_requested_split(self) -> None:
        with TemporaryDirectory() as directory:
            cache_path = Path(directory) / "cache"
            DatasetDict(train=_dataset(1)).save_to_disk(cache_path)

            with self.assertRaisesRegex(KeyError, "validation"):
                load_splits(_settings(str(cache_path)), do_train=False, do_eval=True)


class CollatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.collator = AudioClassificationCollator(
            _settings("unused", input_column="waveform", label_column="target")
        )

    def test_collates_single_channel_waveforms_without_flattening(self) -> None:
        batch = self.collator(
            [
                {"waveform": [1, 2, 3], "target": 0},
                {"waveform": torch.tensor([4.0, 5.0, 6.0]), "target": 1},
            ]
        )

        self.assertEqual(tuple(batch["input_values"].shape), (2, 3))
        self.assertEqual(batch["input_values"].dtype, torch.float32)
        self.assertEqual(batch["labels"].tolist(), [0, 1])
        self.assertEqual(batch["labels"].dtype, torch.int64)

    def test_collator_rejects_multichannel_waveforms(self) -> None:
        with self.assertRaisesRegex(ValueError, r"single-channel.*shape \(2, 2\)"):
            self.collator([{"waveform": [[1, 2], [3, 4]], "target": 0}])

    def test_collator_rejects_variable_lengths(self) -> None:
        with self.assertRaisesRegex(ValueError, r"fixed-length.*\[2, 3\]"):
            self.collator(
                [
                    {"waveform": [1, 2], "target": 0},
                    {"waveform": [3, 4, 5], "target": 1},
                ]
            )

    def test_collator_rejects_empty_batches(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one feature"):
            self.collator([])


class PreprocessingTests(unittest.TestCase):
    def test_disabled_augmentation_leaves_datasets_unchanged(self) -> None:
        train = _dataset(1)
        evaluation = _dataset(1)

        transformed = apply_rawboost_transforms(
            train,
            evaluation,
            augmentation=AugmentationSettings(algorithm=0),
            data=_settings("unused"),
        )

        self.assertIs(transformed[0], train)
        self.assertIs(transformed[1], evaluation)

    def test_original_rawboost_runs_as_dataset_transform(self) -> None:
        train = _dataset(1)
        evaluation = _dataset(1)
        augmented = np.array([10.0, 11.0], dtype=np.float32)

        with patch(
            "utilis.rawboost_utils.process_Rawboost_feature",
            return_value=augmented,
        ) as process:
            transformed_train, transformed_eval = apply_rawboost_transforms(
                train,
                evaluation,
                augmentation=AugmentationSettings(
                    algorithm=4,
                    apply_to_validation=False,
                ),
                data=_settings("unused"),
            )
            item = transformed_train[0]

        process.assert_called_once()
        self.assertEqual(item["input_values"].tolist(), augmented.tolist())
        self.assertEqual(item["labels"], 0)
        self.assertIs(transformed_eval, evaluation)

    def test_all_original_rawboost_recipes_are_configurable(self) -> None:
        for algorithm in range(9):
            with self.subTest(algorithm=algorithm):
                self.assertEqual(AugmentationSettings(algorithm=algorithm).algorithm, algorithm)

    def test_rawboost_algorithm_rejects_bool_and_out_of_range_values(self) -> None:
        for algorithm in (True, -1, 9):
            with self.subTest(algorithm=algorithm), self.assertRaises(ValueError):
                AugmentationSettings(algorithm=algorithm)


if __name__ == "__main__":
    unittest.main()
