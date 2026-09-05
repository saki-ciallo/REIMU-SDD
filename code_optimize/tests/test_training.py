from __future__ import annotations

import unittest
from pathlib import Path

import torch

from add_system.configuration import LossSettings, TrainingSettings, resolve_experiment
from add_system.training.arguments import build_training_arguments
from add_system.training.losses import build_loss


class LossTests(unittest.TestCase):
    def test_relative_output_root_is_anchored_to_optimized_package(self) -> None:
        root = Path(__file__).resolve().parents[1]
        settings = resolve_experiment([root / "configs/common.yaml"]).build_settings()

        self.assertEqual(
            settings.output_dir,
            root / "outputs" / "asv19_baseline",
        )

    def test_cross_entropy_and_focal_are_finite(self) -> None:
        logits = torch.tensor([[2.0, -1.0], [-0.5, 1.0]], requires_grad=True)
        labels = torch.tensor([0, 1])
        for loss_type in ("cross_entropy", "focal"):
            with self.subTest(loss_type=loss_type):
                loss = build_loss(
                    LossSettings(loss_type=loss_type),
                    num_labels=2,
                )(logits, labels)
                self.assertTrue(torch.isfinite(loss))
                self.assertGreater(loss.item(), 0.0)

    def test_structured_training_arguments_use_float_warmup_steps(self) -> None:
        root = Path(__file__).resolve().parents[1]
        resolved = resolve_experiment([root / "configs/common.yaml", root / "configs/smoke.yaml"])
        settings = resolved.build_settings()
        arguments = build_training_arguments(settings)

        self.assertEqual(arguments.max_steps, 2)
        self.assertEqual(arguments.warmup_steps, 0.05)
        self.assertTrue(arguments.remove_unused_columns)
        self.assertEqual(arguments.logging_steps, 1)
        self.assertEqual(settings.data.max_train_samples, 4)

    def test_training_settings_reject_unsupported_fp32_execution(self) -> None:
        with self.assertRaisesRegex(ValueError, "bf16 must be true"):
            TrainingSettings(bf16=False)

    def test_training_strategies_are_normalized(self) -> None:
        settings = TrainingSettings(
            eval_strategy="EPOCH",
            save_strategy="EPOCH",
            logging_strategy="STEPS",
        )

        self.assertEqual(settings.eval_strategy, "epoch")
        self.assertEqual(settings.save_strategy, "epoch")
        self.assertEqual(settings.logging_strategy, "steps")


if __name__ == "__main__":
    unittest.main()
