from __future__ import annotations

from dataclasses import dataclass

import torch
from fla.models.utils import Cache
from torch import nn

type LayerCache = Cache | list[torch.Tensor] | None


@dataclass(slots=True)
class ArchitectureOutput:
    last_hidden_state: torch.Tensor
    past_key_values: LayerCache = None
    hidden_states: tuple[torch.Tensor, ...] | None = None
    attentions: tuple[object | None, ...] | None = None
    aux_loss: torch.Tensor | None = None


def mean_aux_loss(losses: list[torch.Tensor]) -> torch.Tensor | None:
    if not losses:
        return None
    return torch.stack([loss.float() for loss in losses]).mean()


def extend_optional_tuple[T](
    current: tuple[T, ...] | None,
    values: tuple[T, ...] | None,
) -> tuple[T, ...] | None:
    if current is None or values is None:
        return current
    return (*current, *values)


def bridge_input_gradient(
    recurrent_state: torch.Tensor,
    original_input: torch.Tensor,
) -> torch.Tensor:
    """Preserve forward values while reconnecting truncated BPTT to the frontend."""

    return recurrent_state + (original_input - original_input.detach())


class BackboneArchitecture(nn.Module):
    supports_cache = False
