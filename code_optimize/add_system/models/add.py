from __future__ import annotations

from typing import ClassVar

import torch
from fla.models.utils import Cache
from transformers import PreTrainedModel

from ..configuration.blocks import PipelineType
from ..configuration.components.add import ADDConfig
from .aasist import AASISTModel
from .backbones import BackboneModel
from .classifier import ClassifierModel
from .frontends import build_frontend
from .outputs import ADDOutput
from .pooling import PoolingModel


class ADDModel(PreTrainedModel):
    """Waveform frontend followed by one explicitly configured ADD pipeline."""

    config_class = ADDConfig
    base_model_prefix = "model"
    main_input_name = "input_values"
    supports_gradient_checkpointing = True
    all_tied_weights_keys: ClassVar[dict[str, str]] = {}

    def __init__(self, config: ADDConfig) -> None:
        super().__init__(config)
        self.frontend = build_frontend(config.frontend_config)
        if config.model_architecture == PipelineType.BACKBONE_POOLING:
            self.backbone = BackboneModel(config.backbone_config)
            self.pooling = PoolingModel(config.pooling_config)
            self.aasist = None
        else:
            self.backbone = None
            self.pooling = None
            self.aasist = AASISTModel(config.aasist_config)
        self.classifier = ClassifierModel(config.classifier_config)

    def forward(
        self,
        input_values: torch.Tensor,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | list[torch.Tensor] | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple | ADDOutput:
        if input_values.ndim != 2:
            raise ValueError("input_values must have shape [batch, samples].")
        use_return_dict = self.config.return_dict if return_dict is None else return_dict
        hidden_states = self.frontend(input_values=input_values).hidden_state

        if self.config.model_architecture == PipelineType.SSL_AASIST:
            if use_cache or past_key_values is not None:
                raise ValueError("ssl_aasist does not support cache.")
            if output_attentions:
                raise ValueError("ssl_aasist does not return attention maps.")
            aasist_output = self.aasist(input_ids=hidden_states)
            pooled = aasist_output.last_hidden_state
            classifier_output = self.classifier(input_ids=pooled, labels=labels)
            logits = classifier_output.logits
            if not use_return_dict:
                return (
                    logits,
                    pooled,
                    aasist_output.temporal_hidden_state,
                    aasist_output.spectral_hidden_state,
                    aasist_output.master_hidden_state,
                )
            return ADDOutput(
                logits=logits,
                loss_logits=classifier_output.loss_logits,
                pooled_output=pooled,
                temporal_hidden_state=aasist_output.temporal_hidden_state,
                spectral_hidden_state=aasist_output.spectral_hidden_state,
                master_hidden_state=aasist_output.master_hidden_state,
            )

        backbone_output = self.backbone(
            input_ids=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        pooled = self.pooling(input_ids=backbone_output.last_hidden_state).pooled_output
        classifier_output = self.classifier(input_ids=pooled, labels=labels)
        logits = classifier_output.logits
        if not use_return_dict:
            return tuple(
                item
                for item in (
                    logits,
                    pooled,
                    backbone_output.past_key_values,
                    backbone_output.hidden_states,
                    backbone_output.attentions,
                    backbone_output.aux_loss,
                )
                if item is not None
            )
        return ADDOutput(
            logits=logits,
            loss_logits=classifier_output.loss_logits,
            pooled_output=pooled,
            past_key_values=backbone_output.past_key_values,
            hidden_states=backbone_output.hidden_states,
            attentions=backbone_output.attentions,
            aux_loss=backbone_output.aux_loss,
        )
