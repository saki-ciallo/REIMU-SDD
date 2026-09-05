from __future__ import annotations

from fla.modules import GatedMLP
from torch import nn

from ....configuration.blocks import FeedForwardType
from ....configuration.components.backbone import AttentionBackboneConfig
from .moe import LatentMoE


def build_feed_forward(
    feed_forward_type: FeedForwardType,
    config: AttentionBackboneConfig,
) -> nn.Module:
    if feed_forward_type is FeedForwardType.MLP:
        return GatedMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=config.mlp_hidden_ratio,
            intermediate_size=config.mlp_intermediate_size,
            hidden_act=config.mlp_hidden_act,
            fuse_swiglu=config.mlp_fuse_swiglu,
        )
    return LatentMoE(
        hidden_size=config.hidden_size,
        latent_hidden_size=config.resolved_moe_latent_hidden_size,
        latent_intermediate_size=config.resolved_moe_latent_intermediate_size,
        shared_intermediate_size=config.mlp_intermediate_size,
        num_experts=config.moe_num_experts,
        top_k=config.moe_top_k,
        bias=config.bias,
        capacity_factor=config.moe_capacity_factor,
        drop_tokens=config.moe_drop_tokens,
        aux_loss_coefficient=config.moe_aux_loss_coeff,
        use_sequence_aux_loss=config.moe_use_seq_aux_loss,
        use_shared_expert=config.moe_use_shared_expert,
        return_aux_loss=config.return_aux_loss,
        mlp_hidden_ratio=config.mlp_hidden_ratio,
        mlp_hidden_act=config.mlp_hidden_act,
        mlp_fuse_swiglu=config.mlp_fuse_swiglu,
    )
