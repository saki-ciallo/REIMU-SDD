from __future__ import annotations

import unittest
from pathlib import Path


class PackageLayoutTests(unittest.TestCase):
    def test_removed_ambiguous_modules_do_not_return(self) -> None:
        package = Path(__file__).resolve().parents[1] / "add_system"
        removed = (
            package / "config",
            package / "training" / "data.py",
            package / "training" / "augmentation",
            package / "data" / "augmentation",
            package / "models" / "aasist" / "model.py",
            package / "models" / "backbones" / "model.py",
            package / "models" / "backbones" / "stack.py",
            package / "models" / "classifier" / "model.py",
            package / "models" / "pooling" / "model.py",
        )

        present = [str(path.relative_to(package)) for path in removed if path.exists()]
        self.assertEqual(present, [], f"Ambiguous legacy modules remain: {present}")

    def test_configuration_and_data_boundaries_exist(self) -> None:
        package = Path(__file__).resolve().parents[1] / "add_system"
        required = (
            package / "configuration" / "components",
            package / "configuration" / "experiments.py",
            package / "configuration" / "settings.py",
            package / "data" / "collation.py",
            package / "data" / "datasets.py",
            package / "data" / "preprocessing.py",
        )

        missing = [str(path.relative_to(package)) for path in required if not path.exists()]
        self.assertEqual(missing, [], f"Required package boundaries are missing: {missing}")

    def test_bundled_cpu_rawboost_boundaries_exist(self) -> None:
        package = Path(__file__).resolve().parents[1] / "utilis"
        required = (
            package / "__init__.py",
            package / "RawBoost.py",
            package / "rawboost_utils.py",
        )

        missing = [str(path.relative_to(package)) for path in required if not path.exists()]
        self.assertEqual(missing, [], f"Bundled RawBoost files are missing: {missing}")

    def test_lifecycle_entrypoints_exist(self) -> None:
        root = Path(__file__).resolve().parents[1]
        required = (
            root / "train.py",
            root / "train.sh",
            root / "test.sh",
            root / "score.sh",
            root / "run_pipeline.sh",
        )
        missing = [str(path.relative_to(root)) for path in required if not path.exists()]
        self.assertEqual(missing, [], f"Lifecycle entrypoints are missing: {missing}")

    def test_dataset_specific_top_level_entrypoints_do_not_return(self) -> None:
        root = Path(__file__).resolve().parents[1]
        removed = (
            root / "test_asv19.py",
            root / "test_asv19.sh",
            root / "test_asv21.py",
            root / "test_asv21.sh",
            root / "score_asv19.py",
            root / "score_asv19.sh",
            root / "score_asv21.py",
            root / "score_asv21.sh",
        )
        present = [str(path.relative_to(root)) for path in removed if path.exists()]
        self.assertEqual(present, [], f"Dataset-specific top-level entrypoints remain: {present}")

    def test_dataset_evaluation_boundaries_exist(self) -> None:
        root = Path(__file__).resolve().parents[1]
        required = (
            root / "evaluation" / "__init__.py",
            root / "evaluation" / "asv19.py",
            root / "evaluation" / "asv19_test.py",
            root / "evaluation" / "asv19_score.py",
            root / "evaluation" / "asv21.py",
            root / "evaluation" / "asv21_test.py",
            root / "evaluation" / "asv21_score.py",
        )
        missing = [str(path.relative_to(root)) for path in required if not path.exists()]
        self.assertEqual(missing, [], f"Dataset evaluation files are missing: {missing}")


if __name__ == "__main__":
    unittest.main()
