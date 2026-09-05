from __future__ import annotations

import unittest

import torch

from add_system.configuration import (
    AttentionBackboneConfig,
    AttentionFLAConfig,
    GatedDelta2FLAConfig,
    Mamba3FLAConfig,
    RavenFLAConfig,
)
from add_system.models.backbones import BackboneModel


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required by FLA kernels.")
class CUDABackboneGradientTests(unittest.TestCase):
    def test_all_architectures_propagate_finite_input_gradients(self) -> None:
        for architecture in (
            "baseline",
            "looped",
            "hrm",
            "heterogeneous_hrm",
        ):
            with self.subTest(architecture=architecture):
                block = [{"attn": "attention", "mlp": "mlp", "num_layers": 1}]
                block_arguments = (
                    {
                        "blocks": None,
                        "hrm_h_blocks": block,
                        "hrm_l_blocks": block,
                    }
                    if architecture == "heterogeneous_hrm"
                    else {"blocks": block}
                )
                config = AttentionBackboneConfig(
                    hidden_size=128,
                    architecture_type=architecture,
                    attention_config=AttentionFLAConfig(
                        num_heads=4,
                        num_kv_heads=4,
                    ),
                    mlp_intermediate_size=256,
                    looped_num_cycles=2,
                    looped_num_gradient_cycles=1,
                    hrm_h_cycles=2,
                    hrm_l_cycles=1,
                    hrm_num_gradient_steps=2,
                    **block_arguments,
                )
                model = BackboneModel(config).cuda().to(torch.bfloat16).train()
                hidden_states = torch.randn(
                    1,
                    8,
                    128,
                    device="cuda",
                    dtype=torch.bfloat16,
                    requires_grad=True,
                )
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = model(hidden_states, use_cache=False)
                    loss = output.last_hidden_state.float().square().mean()
                loss.backward()

                self.assertTrue(torch.isfinite(hidden_states.grad).all())
                self.assertGreater(hidden_states.grad.float().norm().item(), 0.0)
                parameters = [
                    parameter for parameter in model.parameters() if parameter.requires_grad
                ]
                self.assertTrue(all(parameter.grad is not None for parameter in parameters))

    def test_baseline_cache_accumulates_sequence_length(self) -> None:
        config = AttentionBackboneConfig(
            hidden_size=128,
            architecture_type="baseline",
            blocks=[{"attn": "attention", "mlp": "mlp", "num_layers": 2}],
            attention_config=AttentionFLAConfig(
                num_heads=4,
                num_kv_heads=4,
            ),
            mlp_intermediate_size=256,
            use_cache=True,
        )
        model = BackboneModel(config).cuda().to(torch.bfloat16).eval()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            first = model(
                torch.randn(1, 8, 128, device="cuda", dtype=torch.bfloat16),
                use_cache=True,
                output_attentions=True,
                output_hidden_states=True,
            )
            second = model(
                torch.randn(1, 4, 128, device="cuda", dtype=torch.bfloat16),
                past_key_values=first.past_key_values,
                use_cache=True,
            )

        self.assertEqual(first.last_hidden_state.shape, (1, 8, 128))
        self.assertEqual(second.last_hidden_state.shape, (1, 4, 128))
        self.assertEqual(len(first.past_key_values), 2)
        self.assertEqual(first.past_key_values.get_seq_length(), 12)
        self.assertEqual(len(first.hidden_states), 3)
        self.assertEqual(first.attentions, (None, None))
        self.assertTrue(torch.isfinite(second.last_hidden_state).all())

    def test_all_fla_mixers_backpropagate_finite_gradients(self) -> None:
        mixer_configs = {
            "attention": {
                "attention_config": AttentionFLAConfig(
                    num_heads=4,
                    num_kv_heads=4,
                )
            },
            "raven": {
                "raven_config": RavenFLAConfig(
                    num_heads=4,
                    num_kv_heads=2,
                    topk=4,
                )
            },
            "gdn2": {
                "gdn2_config": GatedDelta2FLAConfig(
                    num_heads=4,
                    num_v_heads=4,
                    head_dim=32,
                )
            },
            "mamba3": {
                "mamba3_config": Mamba3FLAConfig(
                    head_dim=32,
                    state_size=64,
                )
            },
        }
        for mixer_type, mixer_arguments in mixer_configs.items():
            with self.subTest(mixer_type=mixer_type):
                config = AttentionBackboneConfig(
                    hidden_size=128,
                    architecture_type="baseline",
                    blocks=[
                        {
                            "attn": mixer_type,
                            "mlp": "mlp",
                            "num_layers": 1,
                        }
                    ],
                    mlp_intermediate_size=256,
                    **mixer_arguments,
                )
                model = BackboneModel(config).cuda().to(torch.bfloat16).train()
                hidden_states = torch.randn(
                    1,
                    16,
                    128,
                    device="cuda",
                    dtype=torch.bfloat16,
                    requires_grad=True,
                )
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = model(hidden_states, use_cache=False)
                    loss = output.last_hidden_state.float().square().mean()
                loss.backward()

                gradients = [
                    parameter.grad
                    for parameter in model.parameters()
                    if parameter.requires_grad and parameter.grad is not None
                ]
                self.assertTrue(gradients)
                self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
                self.assertGreater(hidden_states.grad.float().norm().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
