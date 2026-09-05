from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoModel

from .backbone_FFN import (
    LatentMoE,
    MixerOutput,
    SwiGLUExperts,
    TopKRouter,
    check_grouped_mm_equivalence,
)
from .backbone_FLA import Attention, GatedDeltaNet2, Mamba3, Raven
from .backbone_Mixer import (
    GDN2Block,
    GDN2MoEBlock,
    Mamba3Block,
    Mamba3MoEBlock,
    RavenBlock,
    RavenMoEBlock,
    SequenceFeedForwardBlock,
    StandardAttentionBlock,
    StandardAttentionMoEBlock,
)
from .classifier import AMSoftmaxClassifier, LinearClassifier
from .configuration import (
    ADDConfig,
    AASISTConfig,
    AttentionBackboneConfig,
    AttentionFLAConfig,
    GatedAttentionPoolingConfig,
    GatedDelta2FLAConfig,
    LinearClassifierConfig,
    LinearFrontendConfig,
    Mamba3FLAConfig,
    RavenFLAConfig,
    SincNetFrontendConfig,
    SSLFrontendConfig,
)
from .frontend import (
    CausalCNN,
    LinearAudioFrontend,
    MixedCNN,
    SincConv_fast,
    StreamingMixedCNNState,
    StreamingSincState,
)
from .aasist import AASISTEncoder
from .modeling_add import ADDModel, ADDOutput
from .modeling_aasist import AASISTModel, AASISTOutput, AASISTSummaryMixin
from .modeling_backbone import (
    AttentionBackboneModel,
    AttentionBackboneOutput,
    AttentionBackbonePreTrainedModel,
    AttentionBackboneSummaryMixin,
)
from .modeling_classifier import (
    LinearClassifierModel,
    LinearClassifierOutput,
    LinearClassifierSummaryMixin,
)
from .modeling_linear import (
    LinearFrontendModel,
    LinearFrontendOutput,
    LinearFrontendSummaryMixin,
)
from .modeling_pooling import (
    GatedAttentionPoolingModel,
    GatedAttentionPoolingOutput,
    GatedAttentionPoolingSummaryMixin,
)
from .modeling_sincnet import (
    SincNetBlock,
    SincNetBlockStreamingState,
    SincNetFrontendModel,
    SincNetFrontendOutput,
    SincNetFrontendSummaryMixin,
)
from .modeling_ssl import (
    SSLFrontendModel,
    SSLFrontendOutput,
    SSLFrontendSummaryMixin,
)
from .pooling import MultiHeadGatedAttentionPooling
from .registration import (
    register_add_for_auto,
    register_aasist_for_auto,
    register_audio_models_for_auto,
    register_classifier_for_auto,
    register_linear_frontend_for_auto,
    register_pooling_for_auto,
    register_sincnet_frontend_for_auto,
    register_ssl_frontend_for_auto,
)


def build_sincnet_random_audio(
    batch_size: int = 4,
    num_samples: int = 16_000,
    seed: int = 0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.randn(batch_size, num_samples, generator=generator, device=device)


def run_sincnet_frontend_shape_demo(
    batch_size: int = 4,
    sampling_rate: int = 16_000,
    seconds: float = 1.0,
    num_filters: int = 80,
    seed: int = 0,
) -> SincNetFrontendOutput:
    register_sincnet_frontend_for_auto()

    num_input_samples = int(round(sampling_rate * seconds))
    input_samples = build_sincnet_random_audio(batch_size=batch_size, num_samples=num_input_samples, seed=seed)

    config = SincNetFrontendConfig(
        target_sampling_rate=sampling_rate,
        window_ms=25.0,
        stride_ms=20.0,
        num_filters=num_filters,
    )
    sincnet_frontend_model = AutoModel.from_config(config)
    outputs = sincnet_frontend_model(input_values=input_samples)
    sinc_features = sincnet_frontend_model.frontend.sincnet(input_samples)

    print(f"input_samples shape: {tuple(input_samples.shape)}")
    print(f"sinc_features shape: {tuple(sinc_features.shape)}")
    print(f"hidden_state shape: {tuple(outputs.hidden_state.shape)}")
    print(f"requested_window_size samples: {config.window_size}")
    print(f"effective_sinc_kernel_size samples: {sincnet_frontend_model.frontend.sincnet.kernel_size}")
    print(f"sinc_stride_size samples: {config.stride_size}")
    print(f"cnn_kernels: {config.cnn_kernels}")
    print(f"cnn_channels: {config.cnn_channels}")
    print(f"cnn_bias: {config.cnn_bias}")
    print(f"apply_final_cnn_activation: {config.apply_final_cnn_activation}")
    print(f"use_final_sinc_residual: {config.use_final_sinc_residual}")
    print(f"final_dropout: {config.final_dropout}")
    print("sinc trainable parameters:")
    for item in sincnet_frontend_model.sinc_parameter_summary():
        print(
            f"  {item['name']}: shape={item['shape']}, dtype={item['dtype']}, "
            f"numel={item['numel']}, meaning={item['meaning']}"
        )
    print("cnn trainable parameters:")
    for layer_item in sincnet_frontend_model.cnns_parameter_summary():
        print(
            f"  {layer_item['layer_name']}: {layer_item['module']}("
            f"{layer_item['input_dim']}->{layer_item['output_dim']}, "
            f"kernel={layer_item['kernel_size']}), numel={layer_item['numel']}"
        )
        for parameter_item in layer_item["parameters"]:
            print(
                f"    {parameter_item['name']}: shape={parameter_item['shape']}, "
                f"dtype={parameter_item['dtype']}, numel={parameter_item['numel']}, "
                f"meaning={parameter_item['meaning']}"
            )
    print(f"total_trainable_parameters: {sincnet_frontend_model.num_trainable_parameters()}")
    return outputs


def analyze_sincnet_frontend(
    batch_size: int = 4,
    sampling_rate: int = 16_000,
    seconds: float = 1.0,
    num_filters: int = 80,
    seed: int = 0,
) -> SincNetFrontendOutput:
    register_sincnet_frontend_for_auto()
    num_input_samples = int(round(sampling_rate * seconds))
    input_samples = build_sincnet_random_audio(batch_size=batch_size, num_samples=num_input_samples, seed=seed)
    config = SincNetFrontendConfig(target_sampling_rate=sampling_rate, num_filters=num_filters)
    sincnet_frontend_model = AutoModel.from_config(config)
    outputs = sincnet_frontend_model(input_values=input_samples)
    sinc_features = sincnet_frontend_model.frontend.sincnet(input_samples)

    print(sincnet_frontend_model.architecture_summary())
    print(f"input_samples shape: {tuple(input_samples.shape)}")
    print(f"sinc_features shape: {tuple(sinc_features.shape)}")
    print(f"hidden_state shape: {tuple(outputs.hidden_state.shape)}")
    print("parameter paths:")
    for item in sincnet_frontend_model.parameter_name_summary():
        print(f"  {item['name']}: shape={item['shape']}, dtype={item['dtype']}, requires_grad={item['requires_grad']}")
    return outputs


__all__ = [
    "ADDConfig",
    "ADDModel",
    "ADDOutput",
    "AASISTConfig",
    "AASISTEncoder",
    "AASISTModel",
    "AASISTOutput",
    "AASISTSummaryMixin",
    "Attention",
    "AttentionBackboneConfig",
    "AttentionBackboneModel",
    "AttentionBackboneOutput",
    "AttentionBackbonePreTrainedModel",
    "AttentionBackboneSummaryMixin",
    "AttentionFLAConfig",
    "AMSoftmaxClassifier",
    "CausalCNN",
    "GDN2Block",
    "GDN2MoEBlock",
    "GatedDeltaNet2",
    "GatedDelta2FLAConfig",
    "GatedAttentionPoolingConfig",
    "GatedAttentionPoolingModel",
    "GatedAttentionPoolingOutput",
    "GatedAttentionPoolingSummaryMixin",
    "LatentMoE",
    "LinearClassifier",
    "LinearClassifierConfig",
    "LinearClassifierModel",
    "LinearClassifierOutput",
    "LinearClassifierSummaryMixin",
    "LinearAudioFrontend",
    "LinearFrontendConfig",
    "LinearFrontendModel",
    "LinearFrontendOutput",
    "LinearFrontendSummaryMixin",
    "Mamba3",
    "Mamba3Block",
    "Mamba3FLAConfig",
    "Mamba3MoEBlock",
    "MixerOutput",
    "MultiHeadGatedAttentionPooling",
    "Raven",
    "RavenBlock",
    "RavenFLAConfig",
    "RavenMoEBlock",
    "SincConv_fast",
    "SincNetBlock",
    "SincNetFrontendConfig",
    "SincNetBlockStreamingState",
    "StreamingSincState",
    "StreamingMixedCNNState",
    "SincNetFrontendModel",
    "SincNetFrontendOutput",
    "SincNetFrontendSummaryMixin",
    "SSLFrontendConfig",
    "SSLFrontendModel",
    "SSLFrontendOutput",
    "SSLFrontendSummaryMixin",
    "SequenceFeedForwardBlock",
    "StandardAttentionBlock",
    "StandardAttentionMoEBlock",
    "SwiGLUExperts",
    "TopKRouter",
    "MixedCNN",
    "analyze_sincnet_frontend",
    "build_sincnet_random_audio",
    "check_grouped_mm_equivalence",
    "register_add_for_auto",
    "register_aasist_for_auto",
    "register_audio_models_for_auto",
    "register_classifier_for_auto",
    "register_linear_frontend_for_auto",
    "register_pooling_for_auto",
    "register_sincnet_frontend_for_auto",
    "register_ssl_frontend_for_auto",
    "run_sincnet_frontend_shape_demo",
]
