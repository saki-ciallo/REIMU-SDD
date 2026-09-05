from __future__ import annotations

import tempfile
import unittest

from add_system.configuration import (
    AASISTConfig,
    ADDConfig,
    AttentionBackboneConfig,
    AttentionFLAConfig,
    GatedAttentionPoolingConfig,
    LinearClassifierConfig,
    SSLFrontendConfig,
)


def _attention_backbone(
    *,
    architecture_type: str = "baseline",
    **kwargs: object,
) -> AttentionBackboneConfig:
    return AttentionBackboneConfig(
        hidden_size=128,
        architecture_type=architecture_type,
        attention_config=AttentionFLAConfig(num_heads=4, num_kv_heads=4),
        **kwargs,
    )


class BackboneConfigTests(unittest.TestCase):
    def test_looped_gradient_schedule_opens_only_tail_cycles(self) -> None:
        config = _attention_backbone(
            architecture_type="looped",
            blocks=[{"attn": "attention", "mlp": "mlp", "num_layers": 2}],
            looped_num_cycles=4,
            looped_num_gradient_cycles=2,
        )

        self.assertEqual(config.looped_gradient_schedule, (False, False, True, True))
        self.assertEqual(config.num_hidden_layers, 2)

    def test_hrm_gradient_schedule_counts_l_and_h_modules(self) -> None:
        config = _attention_backbone(
            architecture_type="hrm",
            blocks=[{"attn": "attention", "mlp": "mlp", "num_layers": 2}],
            hrm_h_cycles=2,
            hrm_l_cycles=3,
            hrm_num_gradient_steps=2,
        )

        self.assertEqual(config.hrm_total_steps, 8)
        self.assertEqual(config.hrm_gradient_schedule, (False,) * 6 + (True, True))

    def test_heterogeneous_hrm_keeps_independent_specs(self) -> None:
        config = _attention_backbone(
            architecture_type="heterogeneous_hrm",
            hrm_h_blocks=[{"attn": "attention", "mlp": "mlp", "num_layers": 3}],
            hrm_l_blocks=[{"attn": "gdn2", "mlp": "mlp", "num_layers": 3}],
        )

        self.assertIsNone(config.blocks)
        self.assertEqual(config.hrm_h_num_hidden_layers, 3)
        self.assertEqual(config.hrm_l_num_hidden_layers, 3)


class ADDConfigTests(unittest.TestCase):
    def test_backbone_pipeline_round_trips_through_hf_checkpoint(self) -> None:
        config = ADDConfig(
            frontend_config=SSLFrontendConfig(
                output_size=128,
                freeze_ssl_model=True,
                ssl_trainable_layer_indices=[10, 11],
            ),
            backbone_config=_attention_backbone(
                blocks=[{"attn": "attention", "mlp": "mlp", "num_layers": 2}]
            ),
            pooling_config=GatedAttentionPoolingConfig(
                input_size=128,
                output_size=64,
                num_heads=4,
            ),
            classifier_config=LinearClassifierConfig(input_size=64),
        )

        with tempfile.TemporaryDirectory() as directory:
            config.save_pretrained(directory)
            restored = ADDConfig.from_pretrained(directory)

        self.assertIsInstance(restored.frontend_config, SSLFrontendConfig)
        self.assertIsInstance(restored.backbone_config, AttentionBackboneConfig)
        self.assertEqual(restored.backbone_config.num_hidden_layers, 2)
        self.assertEqual(restored.model_architecture, "backbone_pooling")
        self.assertIn("pooled_output", restored.keys_to_ignore_at_inference)
        self.assertNotIn("logits", restored.keys_to_ignore_at_inference)

    def test_pipeline_dimension_mismatch_fails_before_model_build(self) -> None:
        with self.assertRaisesRegex(ValueError, "frontend output dimension"):
            ADDConfig(
                frontend_config=SSLFrontendConfig(output_size=256),
                backbone_config=_attention_backbone(
                    blocks=[{"attn": "attention", "mlp": "mlp", "num_layers": 1}]
                ),
                pooling_config=GatedAttentionPoolingConfig(
                    input_size=128,
                    output_size=64,
                ),
                classifier_config=LinearClassifierConfig(input_size=64),
            )

    def test_ssl_aasist_pipeline_validates_embedding_size(self) -> None:
        config = ADDConfig(
            model_architecture="ssl_aasist",
            frontend_config=SSLFrontendConfig(output_size=768),
            aasist_config=AASISTConfig(input_size=768),
            classifier_config=LinearClassifierConfig(input_size=160),
        )

        self.assertEqual(config.architecture_type, "ssl_aasist")


if __name__ == "__main__":
    unittest.main()
