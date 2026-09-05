from __future__ import annotations

import unittest

import torch

from add_system.configuration import (
    AASISTConfig,
    GatedAttentionPoolingConfig,
    LinearClassifierConfig,
    LinearFrontendConfig,
    SincNetFrontendConfig,
)
from add_system.models.aasist import AASISTModel
from add_system.models.classifier import ClassifierModel
from add_system.models.frontends import LinearFrontendModel, SincNetFrontendModel
from add_system.models.pooling import PoolingModel


class FrontendTests(unittest.TestCase):
    def test_linear_frontend_slices_complete_frames(self) -> None:
        model = LinearFrontendModel(LinearFrontendConfig(output_size=32))
        output = model(torch.randn(2, 1_050)).hidden_state

        self.assertEqual(output.shape, (2, 2, 32))

    def test_sincnet_preserves_sequence_and_output_contract(self) -> None:
        config = SincNetFrontendConfig(
            cnn_kernels=[5, 3],
            cnn_channels=[32, 64],
        )
        model = SincNetFrontendModel(config)
        output = model(torch.randn(2, 4_000)).hidden_state

        self.assertEqual(output.ndim, 3)
        self.assertEqual(output.shape[0], 2)
        self.assertEqual(output.shape[-1], 64)
        self.assertTrue(torch.isfinite(output).all())


class PoolingClassifierTests(unittest.TestCase):
    def test_all_pooling_types_backpropagate(self) -> None:
        for pooling_type in ("mhgap", "gamp", "gapv1", "gapv2"):
            with self.subTest(pooling_type=pooling_type):
                hidden_states = torch.randn(2, 12, 32, requires_grad=True)
                pooling = PoolingModel(
                    GatedAttentionPoolingConfig(
                        pooling_type=pooling_type,
                        input_size=32,
                        output_size=16,
                        num_heads=4,
                    )
                )
                classifier = ClassifierModel(LinearClassifierConfig(input_size=16))
                logits = classifier(pooling(hidden_states).pooled_output).logits
                logits.square().mean().backward()

                self.assertEqual(logits.shape, (2, 2))
                self.assertTrue(torch.isfinite(hidden_states.grad).all())
                self.assertGreater(hidden_states.grad.norm().item(), 0.0)

    def test_amsoftmax_separates_prediction_and_margin_logits(self) -> None:
        classifier = ClassifierModel(
            LinearClassifierConfig(
                input_size=16,
                classifier_type="amsoftmax",
                am_margin=0.3,
                am_scale=15.0,
            )
        )
        hidden_states = torch.randn(2, 16)
        labels = torch.tensor([0, 1])

        without_labels = classifier(hidden_states)
        with_labels = classifier(hidden_states, labels=labels)

        torch.testing.assert_close(with_labels.logits, without_labels.logits)
        expected = with_labels.logits.clone()
        expected[torch.arange(2), labels] -= 0.3 * 15.0
        torch.testing.assert_close(with_labels.loss_logits, expected)


class AASISTTests(unittest.TestCase):
    def test_aasist_returns_embedding_and_graph_states(self) -> None:
        model = AASISTModel(AASISTConfig())
        output = model(torch.randn(2, 199, 768))

        self.assertEqual(output.last_hidden_state.shape, (2, 160))
        self.assertEqual(output.temporal_hidden_state.shape[-1], 32)
        self.assertEqual(output.spectral_hidden_state.shape[-1], 32)
        self.assertEqual(output.master_hidden_state.shape, (2, 1, 32))


if __name__ == "__main__":
    unittest.main()
