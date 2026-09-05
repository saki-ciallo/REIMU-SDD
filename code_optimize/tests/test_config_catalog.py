from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from add_system.configuration import resolve_experiment
from add_system.configuration.experiments import deep_merge, load_yaml_mapping
from migrate_configs_to_new import convert_legacy_config, migrate_configs


class ConfigCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.optimized_root = Path(__file__).resolve().parents[1]
        cls.repository_root = cls.optimized_root.parent
        cls.source_root = cls.repository_root / "configs"
        cls.source_common = load_yaml_mapping(cls.source_root / "train_common.yaml")
        cls.source_defaults = {
            "task_name": "baseline",
            "blocks_json": '{"0":{"attn":"raven","mlp":"mlp","num_layers":8}}',
            "hrm_h_blocks_json": None,
            "hrm_l_blocks_json": None,
            "architecture_type": "baseline",
            "looped_num_cycles": 2,
            "looped_num_gradient_cycles": 1,
            "hrm_h_cycles": 2,
            "hrm_l_cycles": 3,
            "hrm_num_gradient_steps": 2,
        }
        cls.resolved_source_common = deep_merge(cls.source_defaults, cls.source_common)

    def test_every_source_experiment_has_an_optimized_equivalent(self) -> None:
        source_paths = sorted((self.source_root / "experiments").rglob("*.yaml"))
        destination_paths = sorted(
            (self.optimized_root / "configs" / "experiments").rglob("*.yaml")
        )
        source_names = {path.relative_to(self.source_root / "experiments") for path in source_paths}
        destination_names = {
            path.relative_to(self.optimized_root / "configs" / "experiments")
            for path in destination_paths
        }

        self.assertEqual(len(source_paths), 163)
        self.assertEqual(destination_names, source_names)

    def test_conversion_rejects_unmapped_legacy_field(self) -> None:
        source = dict(self.resolved_source_common)
        source["new_legacy_field"] = True

        with self.assertRaisesRegex(ValueError, "unmapped legacy fields: new_legacy_field"):
            convert_legacy_config(source)

    def test_migration_rejects_unmapped_overlay_before_writing(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            source_root = temporary_root / "source"
            destination_root = temporary_root / "destination"
            experiments_root = source_root / "experiments"
            experiments_root.mkdir(parents=True)
            (source_root / "train_common.yaml").write_text(
                yaml.safe_dump(self.source_common, sort_keys=False),
                encoding="utf-8",
            )
            source_path = experiments_root / "unknown.yaml"
            source_path.write_text(
                yaml.safe_dump(
                    {"task_name": "unknown", "new_legacy_field": True},
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "new_legacy_field"):
                migrate_configs(source_root, destination_root)

            self.assertFalse(destination_root.exists())

    def test_every_optimized_experiment_matches_source_semantics(self) -> None:
        optimized_common = self.optimized_root / "configs" / "common.yaml"
        for source_path in sorted((self.source_root / "experiments").rglob("*.yaml")):
            relative = source_path.relative_to(self.source_root / "experiments")
            optimized_path = self.optimized_root / "configs" / "experiments" / relative
            with self.subTest(config=str(relative)):
                source = deep_merge(
                    self.resolved_source_common,
                    load_yaml_mapping(source_path),
                )
                expected = convert_legacy_config(source)
                resolved = resolve_experiment([optimized_common, optimized_path])

                self.assertEqual(resolved.values, expected)
                settings = resolved.build_settings()
                self.assertEqual(settings.run.task_name, source["task_name"])
                self.assertEqual(settings.model.model_architecture, source["model_architecture"])
                self.assertEqual(settings.augmentation.algorithm, source["rawboost_algo"])


if __name__ == "__main__":
    unittest.main()
