from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from fla.models.utils import Cache
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from .configuration import (
    ADDConfig,
    LinearFrontendConfig,
    SincNetFrontendConfig,
    SSLFrontendConfig,
)
from .modeling_aasist import AASISTModel
from .modeling_backbone import build_backbone_model
from .modeling_classifier import LinearClassifierModel
from .modeling_linear import LinearFrontendModel
from .modeling_pooling import GatedAttentionPoolingModel
from .modeling_sincnet import SincNetBlockStreamingState, SincNetFrontendModel
from .modeling_ssl import SSLFrontendModel


FrontendConfig = SincNetFrontendConfig | LinearFrontendConfig | SSLFrontendConfig
FrontendModel = SincNetFrontendModel | LinearFrontendModel | SSLFrontendModel


@dataclass
class ADDOutput(ModelOutput):
    logits: Optional[torch.Tensor] = None
    pooled_output: Optional[torch.Tensor] = None
    past_key_values: Optional[Cache | list[torch.Tensor]] = None
    hidden_states: Optional[tuple[torch.Tensor, ...]] = None
    attentions: Optional[tuple[object | None, ...]] = None
    aux_loss: Optional[torch.Tensor] = None
    temporal_hidden_state: Optional[torch.Tensor] = None
    spectral_hidden_state: Optional[torch.Tensor] = None
    master_hidden_state: Optional[torch.Tensor] = None


class ADDModel(PreTrainedModel):
    config_class = ADDConfig
    base_model_prefix = "model"
    main_input_name = "input_values"
    supports_gradient_checkpointing = True
    all_tied_weights_keys = {}

    def __init__(self, config: ADDConfig) -> None:
        super().__init__(config)
        self.frontend = self._make_frontend(config.frontend_config)
        if config.model_architecture == "backbone_pooling":
            self.backbone = build_backbone_model(config.backbone_config)
            self.pooling = GatedAttentionPoolingModel(config.pooling_config)
            self.aasist = None
        else:
            self.backbone = None
            self.pooling = None
            self.aasist = AASISTModel(config.aasist_config)
        self.classifier = LinearClassifierModel(config.classifier_config)
        self._validate_component_shapes()
        self.gradient_checkpointing = False

    @staticmethod
    def _make_frontend(config: FrontendConfig) -> FrontendModel:
        if isinstance(config, SincNetFrontendConfig):
            return SincNetFrontendModel(config)
        if isinstance(config, LinearFrontendConfig):
            return LinearFrontendModel(config)
        if isinstance(config, SSLFrontendConfig):
            return SSLFrontendModel(config)
        raise TypeError(
            "frontend_config must be a SincNetFrontendConfig, "
            "LinearFrontendConfig, or SSLFrontendConfig."
        )

    @staticmethod
    def _frontend_output_dim(frontend: FrontendModel) -> int:
        return int(frontend.config.frontend_output_dim)

    def _validate_component_shapes(self) -> None:
        frontend_output_dim = self._frontend_output_dim(self.frontend)
        classifier_input_size = self.classifier.config.input_size
        if self.config.model_architecture == "backbone_pooling":
            backbone_hidden_size = self.backbone.config.hidden_size
            pooling_input_size = self.pooling.config.input_size
            pooling_output_size = self.pooling.config.output_size
            if frontend_output_dim != backbone_hidden_size:
                raise ValueError(
                    f"frontend output_dim={frontend_output_dim} must match "
                    f"backbone hidden_size={backbone_hidden_size}."
                )
            if pooling_input_size != backbone_hidden_size:
                raise ValueError(
                    f"pooling input_size={pooling_input_size} must match "
                    f"backbone hidden_size={backbone_hidden_size}."
                )
            if classifier_input_size != pooling_output_size:
                raise ValueError(
                    f"classifier input_size={classifier_input_size} must match "
                    f"pooling output_size={pooling_output_size}."
                )
            return

        aasist_input_size = self.aasist.config.input_size
        aasist_output_size = self.aasist.config.output_size
        if frontend_output_dim != aasist_input_size:
            raise ValueError(
                f"SSL frontend output_dim={frontend_output_dim} must match "
                f"AASIST input_size={aasist_input_size}."
            )
        if classifier_input_size != aasist_output_size:
            raise ValueError(
                f"classifier input_size={classifier_input_size} must match "
                f"AASIST output_size={aasist_output_size}."
            )

    def init_streaming_state(self) -> SincNetBlockStreamingState:
        if not isinstance(self.frontend, SincNetFrontendModel):
            raise NotImplementedError("forward_streaming is only supported by SincNetFrontendModel.")
        return self.frontend.init_streaming_state()

    def reset_streaming_state(self) -> SincNetBlockStreamingState:
        return self.init_streaming_state()

    def forward(
        self,
        input_values: torch.Tensor,
        labels: torch.Tensor | None = None, # for Trainer compatibility, not used in the model itself
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache | list[torch.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Unpack[dict],
    ) -> tuple | ADDOutput:
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected keyword arguments: {unexpected}")

        return_dict = return_dict if return_dict is not None else self.config.return_dict
        if use_cache is None:
            use_cache = (
                self.config.backbone_config.use_cache
                if self.config.model_architecture == "backbone_pooling" and not self.training
                else False
            )
        output_attentions = self.config.output_attentions if output_attentions is None else output_attentions
        output_hidden_states = (
            self.config.output_hidden_states
            if output_hidden_states is None
            else output_hidden_states
        )

        if input_values.ndim != 2:
            raise ValueError("input_values must have shape [batch_size, num_samples].")

        # Step1: SincNet, framed Linear, or pretrained SSL frontend.
        frontend_outputs = self.frontend(input_values=input_values)
        hidden_states = frontend_outputs.hidden_state

        if self.config.model_architecture == "ssl_aasist":
            if use_cache or past_key_values is not None:
                raise ValueError("model_architecture='ssl_aasist' does not support cache.")
            if output_attentions:
                raise ValueError(
                    "model_architecture='ssl_aasist' does not return attention maps."
                )
            aasist_outputs = self.aasist(input_ids=hidden_states)
            pooled_output = aasist_outputs.last_hidden_state
            classifier_outputs = self.classifier(input_ids=pooled_output, labels=labels)
            logits = classifier_outputs.logits
            if not return_dict:
                return (
                    logits,
                    pooled_output,
                    aasist_outputs.temporal_hidden_state,
                    aasist_outputs.spectral_hidden_state,
                    aasist_outputs.master_hidden_state,
                )
            return ADDOutput(
                logits=logits,
                pooled_output=pooled_output,
                temporal_hidden_state=aasist_outputs.temporal_hidden_state,
                spectral_hidden_state=aasist_outputs.spectral_hidden_state,
                master_hidden_state=aasist_outputs.master_hidden_state,
            )

        # Step2: Attention Backbone
        backbone_outputs = self.backbone(
            input_ids=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=bool(output_hidden_states),
        )
        hidden_states = backbone_outputs.last_hidden_state # 用于后续的pooling
        past_key_values = backbone_outputs.past_key_values # attendion缓存
        all_hidden_states = backbone_outputs.hidden_states # 每层的hidden states
        all_attns = backbone_outputs.attentions # 仅attention/MHA/GQA才会有返回
        aux_loss = backbone_outputs.aux_loss # MoE可选的辅助损失

        # Step3: Gated Attention Pooling
        pooling_outputs = self.pooling(input_ids=hidden_states)
        pooled_output = pooling_outputs.pooled_output

        # Step4: Classifier
        classifier_outputs = self.classifier(input_ids=pooled_output, labels=labels)
        logits = classifier_outputs.logits

        if not return_dict:
            return tuple(
                item
                for item in (
                    logits,
                    pooled_output,
                    past_key_values,
                    all_hidden_states,
                    all_attns,
                    aux_loss,
                )
                if item is not None
            )
        return ADDOutput(
            logits=logits,
            pooled_output=pooled_output,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attns,
            aux_loss=aux_loss,
        )

    def forward_streaming(
        self,
        new_input_values: torch.Tensor,
        state: Optional[SincNetBlockStreamingState] = None,
        past_key_values: Optional[Cache | list[torch.Tensor]] = None,
        use_cache: Optional[bool] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> tuple | ADDOutput | tuple[None, SincNetBlockStreamingState]:
        if self.config.model_architecture != "backbone_pooling":
            raise NotImplementedError(
                "forward_streaming is only supported by the backbone_pooling pipeline."
            )
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        use_cache = (
            use_cache
            if use_cache is not None
            else (self.config.backbone_config.use_cache if not self.training else False)
        )
        output_attentions = self.config.output_attentions if output_attentions is None else output_attentions
        state = self.reset_streaming_state() if state is None else state
        hidden_states, _, state = self.frontend.forward_streaming(new_input_values, state)
        if hidden_states is None:
            return None, state

        backbone_outputs = self.backbone(
            input_ids=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = backbone_outputs.last_hidden_state
        past_key_values = backbone_outputs.past_key_values
        all_attns = backbone_outputs.attentions
        aux_loss = backbone_outputs.aux_loss

        pooling_outputs = self.pooling(input_ids=hidden_states)
        pooled_output = pooling_outputs.pooled_output

        classifier_outputs = self.classifier(input_ids=pooled_output)
        logits = classifier_outputs.logits

        if not return_dict:
            return (
                tuple(
                    item
                    for item in (
                        logits,
                        pooled_output,
                        past_key_values,
                        all_attns,
                        aux_loss,
                    )
                    if item is not None
                ),
                state,
            )
        return (
            ADDOutput(
                logits=logits,
                pooled_output=pooled_output,
                past_key_values=past_key_values,
                hidden_states=None,
                attentions=all_attns,
                aux_loss=aux_loss,
            ),
            state,
        )

    def parameter_name_summary(self) -> List[Dict[str, object]]:
        return [
            {
                "name": name,
                "shape": tuple(parameter.shape),
                "dtype": str(parameter.dtype).replace("torch.", ""),
                "requires_grad": parameter.requires_grad,
            }
            for name, parameter in self.named_parameters()
        ]

    def num_trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


__all__ = [
    "ADDModel",
    "ADDOutput",
]
