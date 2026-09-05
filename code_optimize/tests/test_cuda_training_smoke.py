from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from add_system.configuration import resolve_experiment
from add_system.models import ADDModel
from add_system.training.losses import build_loss


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required by FLA kernels.")
class CUDATrainingSmokeTests(unittest.TestCase):
    @staticmethod
    def _settings():
        root = Path(__file__).resolve().parents[1]
        return resolve_experiment(
            [root / "configs/common.yaml", root / "configs/smoke.yaml"]
        ).build_settings()

    def test_end_to_end_optimizer_step_updates_parameters(self) -> None:
        settings = self._settings()
        torch.manual_seed(7)
        model = ADDModel(settings.model).cuda().train()
        loss_function = build_loss(
            settings.loss,
            settings.model.classifier_config.num_labels,
        ).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)
        input_values = torch.randn(2, 16_000, device="cuda")
        labels = torch.tensor([0, 1], device="cuda")
        tracked_parameter = next(
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.ndim > 1
        )
        before_step = tracked_parameter.detach().clone()

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(
                input_values=input_values,
                use_cache=False,
                return_dict=True,
            )
            loss = loss_function(outputs.logits.float(), labels)
        loss.backward()

        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        gradient_norm = torch.linalg.vector_norm(
            torch.stack([gradient.detach().float().norm() for gradient in gradients])
        )
        optimizer.step()
        maximum_update = (tracked_parameter.detach() - before_step).float().abs().max()

        self.assertEqual(outputs.logits.shape, (2, 2))
        self.assertEqual(outputs.logits.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(loss.detach()))
        self.assertTrue(torch.isfinite(gradient_norm))
        self.assertGreater(gradient_norm.item(), 0.0)
        self.assertGreater(maximum_update.item(), 0.0)

    def test_checkpoint_round_trip_preserves_logits(self) -> None:
        settings = self._settings()
        torch.manual_seed(11)
        model = ADDModel(settings.model).cuda().eval()
        input_values = torch.randn(1, 16_000, device="cuda")

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            expected = model(
                input_values=input_values,
                use_cache=False,
                return_dict=True,
            ).logits
        with TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            restored = ADDModel.from_pretrained(directory).cuda().eval()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                actual = restored(
                    input_values=input_values,
                    use_cache=False,
                    return_dict=True,
                ).logits

        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
