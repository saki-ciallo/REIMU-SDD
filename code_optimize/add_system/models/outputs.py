from __future__ import annotations

from dataclasses import dataclass

import torch
from fla.models.utils import Cache
from transformers.utils import ModelOutput


@dataclass
class FrontendOutput(ModelOutput):
    hidden_state: torch.Tensor | None = None


@dataclass
class PoolingOutput(ModelOutput):
    pooled_output: torch.Tensor | None = None


@dataclass
class ClassifierOutput(ModelOutput):
    logits: torch.Tensor | None = None
    loss_logits: torch.Tensor | None = None


@dataclass
class BackboneOutput(ModelOutput):
    last_hidden_state: torch.Tensor | None = None
    past_key_values: Cache | list[torch.Tensor] | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None
    attentions: tuple[object | None, ...] | None = None
    aux_loss: torch.Tensor | None = None


@dataclass
class AASISTOutput(ModelOutput):
    last_hidden_state: torch.Tensor | None = None
    temporal_hidden_state: torch.Tensor | None = None
    spectral_hidden_state: torch.Tensor | None = None
    master_hidden_state: torch.Tensor | None = None


@dataclass
class ADDOutput(ModelOutput):
    logits: torch.Tensor | None = None
    loss_logits: torch.Tensor | None = None
    pooled_output: torch.Tensor | None = None
    past_key_values: Cache | list[torch.Tensor] | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None
    attentions: tuple[object | None, ...] | None = None
    aux_loss: torch.Tensor | None = None
    temporal_hidden_state: torch.Tensor | None = None
    spectral_hidden_state: torch.Tensor | None = None
    master_hidden_state: torch.Tensor | None = None
