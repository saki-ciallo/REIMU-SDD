from __future__ import annotations

import unittest

import torch

from add_system.configuration import AttentionBackboneConfig, AttentionFLAConfig
from add_system.models.backbones import BackboneModel


@unittest.skipUnless(torch.cuda.is_available(), "LatentMoE grouped MM requires CUDA.")
class CudaLatentMoETests(unittest.TestCase):
    def test_grouped_experts_and_aux_loss_backpropagate(self) -> None:
        config = AttentionBackboneConfig(
            hidden_size=128,
            architecture_type="baseline",
            blocks=[{"attn": "attention", "mlp": "moe", "num_layers": 1}],
            attention_config=AttentionFLAConfig(
                num_heads=4,
                num_kv_heads=4,
            ),
            mlp_intermediate_size=256,
            moe_latent_hidden_size=64,
            moe_latent_intermediate_size=96,
            moe_num_experts=4,
            moe_top_k=2,
            moe_aux_loss_coeff=0.01,
            moe_use_seq_aux_loss=True,
            return_aux_loss=True,
        )
        model = BackboneModel(config).cuda().to(torch.bfloat16).train()
        hidden_states = torch.randn(
            2,
            8,
            128,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(hidden_states, use_cache=False)
            loss = output.last_hidden_state.float().square().mean() + output.aux_loss
        loss.backward()

        router = model.architecture.stack.layers[0].mlp.router
        experts = model.architecture.stack.layers[0].mlp.experts
        self.assertIsNotNone(output.aux_loss)
        self.assertTrue(torch.isfinite(output.aux_loss))
        self.assertGreater(output.aux_loss.item(), 0.0)
        self.assertTrue(torch.isfinite(router.weight.grad).all())
        self.assertGreater(router.weight.grad.float().norm().item(), 0.0)
        self.assertTrue(torch.isfinite(experts.gate_up_projection.grad).all())
        self.assertGreater(
            experts.gate_up_projection.grad.float().norm().item(),
            0.0,
        )
        self.assertTrue(torch.isfinite(hidden_states.grad).all())


if __name__ == "__main__":
    unittest.main()
