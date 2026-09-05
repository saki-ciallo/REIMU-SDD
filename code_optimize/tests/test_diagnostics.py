from __future__ import annotations

import unittest
from pathlib import Path

from add_system.configuration import resolve_experiment
from add_system.diagnostics import build_model_report
from add_system.models import ADDModel


class ModelReportTests(unittest.TestCase):
    def test_report_names_concrete_pipeline_components(self) -> None:
        root = Path(__file__).resolve().parents[1]
        settings = resolve_experiment(
            [root / "configs/common.yaml", root / "configs/smoke.yaml"]
        ).build_settings()
        report = build_model_report(ADDModel(settings.model))

        self.assertIn("`LinearFrontendModel`", report)
        self.assertIn("`BaselineArchitecture`", report)
        self.assertIn("`Attention`", report)
        self.assertIn("`GatedMLP`", report)
        self.assertIn("`MultiHeadGatedAttentionPooling`", report)
        self.assertIn("`Linear`", report)


if __name__ == "__main__":
    unittest.main()
