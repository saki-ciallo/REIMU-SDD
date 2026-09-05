from __future__ import annotations

from fla.layers import GatedDeltaNet2, Mamba3, Raven
from fla.layers.attn import Attention
from torch import nn

from ....configuration.blocks import MixerType
from ....configuration.components.backbone import AttentionBackboneConfig


def build_mixer(
    mixer_type: MixerType,
    config: AttentionBackboneConfig,
    layer_index: int,
) -> nn.Module:
    match mixer_type:
        case MixerType.ATTENTION:
            mixer_config = config.attention_config
            return Attention(
                hidden_size=config.hidden_size,
                num_heads=mixer_config.num_heads,
                num_kv_heads=mixer_config.num_kv_heads,
                qkv_bias=config.bias,
                qk_norm=mixer_config.qk_norm,
                window_size=mixer_config.window_size,
                rope_theta=mixer_config.rope_theta,
                max_position_embeddings=mixer_config.max_position_embeddings,
                layer_idx=layer_index,
            )
        case MixerType.RAVEN:
            mixer_config = config.raven_config
            return Raven(
                hidden_size=config.hidden_size,
                num_heads=mixer_config.num_heads,
                num_kv_heads=mixer_config.num_kv_heads,
                num_slots=mixer_config.num_slots,
                topk=mixer_config.topk,
                norm_eps=config.norm_eps,
                feature_map=mixer_config.feature_map,
                decay_type=mixer_config.decay_type,
                router_score=mixer_config.router_score,
                router_type=mixer_config.router_type,
                layer_idx=layer_index,
            )
        case MixerType.GDN2:
            mixer_config = config.gdn2_config
            return GatedDeltaNet2(
                hidden_size=config.hidden_size,
                num_heads=mixer_config.num_heads,
                num_v_heads=mixer_config.num_v_heads,
                head_dim=mixer_config.head_dim,
                norm_eps=config.norm_eps,
                layer_idx=layer_index,
            )
        case MixerType.MAMBA3:
            if not hasattr(Mamba3, "_compute_a"):
                raise RuntimeError("Installed FLA Mamba3 lacks _compute_a; upgrade FLA before use.")
            mixer_config = config.mamba3_config
            return Mamba3(
                hidden_size=config.hidden_size,
                head_dim=mixer_config.head_dim,
                state_size=mixer_config.state_size,
                is_mimo=mixer_config.is_mimo,
                use_bias=config.bias,
                norm_eps=config.norm_eps,
                layer_idx=layer_index,
            )
