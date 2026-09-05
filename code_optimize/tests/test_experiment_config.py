from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from add_system.configuration import AttentionBackboneConfig
from add_system.configuration.experiments import (
    deep_merge,
    load_yaml_mapping,
    resolve_experiment,
)


class ExperimentConfigTests(unittest.TestCase):
    def test_deep_merge_preserves_sibling_values(self) -> None:
        merged = deep_merge(
            {"model": {"hidden_size": 128, "dropout": 0.1}, "seed": 42},
            {"model": {"dropout": 0.2}},
        )
        self.assertEqual(
            merged,
            {
                "model": {"hidden_size": 128, "dropout": 0.2},
                "seed": 42,
            },
        )

    def test_deep_merge_replaces_component_when_model_type_changes(self) -> None:
        merged = deep_merge(
            {
                "frontend": {
                    "model_type": "ssl_frontend",
                    "ssl_type": "wav2vec2-base",
                }
            },
            {
                "frontend": {
                    "model_type": "linear_frontend",
                    "output_size": 128,
                }
            },
        )

        self.assertEqual(
            merged["frontend"],
            {"model_type": "linear_frontend", "output_size": 128},
        )

    def test_duplicate_yaml_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.yaml"
            path.write_text("seed: 1\nseed: 2\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Duplicate YAML key"):
                load_yaml_mapping(path)

    def test_resolved_experiment_has_stable_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            common = root / "common.yaml"
            experiment = root / "experiment.yaml"
            common.write_text(
                "model:\n  hidden_size: 128\nseed: 42\n",
                encoding="utf-8",
            )
            experiment.write_text(
                "model:\n  dropout: 0.1\n",
                encoding="utf-8",
            )

            first = resolve_experiment([common, experiment])
            second = resolve_experiment([common, experiment])

            self.assertEqual(
                first.values["model"],
                {"hidden_size": 128, "dropout": 0.1},
            )
            self.assertEqual(first.sha256, second.sha256)
            self.assertEqual(len(first.sha256), 64)

    def test_structured_experiment_builds_typed_model_config(self) -> None:
        root = Path(__file__).resolve().parents[1]
        resolved = resolve_experiment(
            [
                root / "configs/common.yaml",
                root
                / "configs/experiments/looped"
                / "looped_attention_l6_n2_freeze_wav2vec2_mhgap.yaml",
            ]
        )

        model_config = resolved.build_model_config()

        self.assertIsInstance(model_config.backbone_config, AttentionBackboneConfig)
        self.assertEqual(model_config.backbone_config.architecture_type, "looped")
        self.assertEqual(
            model_config.backbone_config.looped_gradient_schedule,
            (False, True),
        )

    def test_unknown_model_field_is_rejected(self) -> None:
        root = Path(__file__).resolve().parents[1]
        resolved = resolve_experiment(
            [root / "configs/common.yaml"],
            overrides={"model": {"backbone_config": {"hidden_sze": 128}}},
        )

        with self.assertRaisesRegex(ValueError, "hidden_sze"):
            resolved.build_model_config()


if __name__ == "__main__":
    unittest.main()
