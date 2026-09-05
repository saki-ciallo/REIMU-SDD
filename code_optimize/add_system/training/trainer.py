from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch import nn
from transformers import EvalPrediction, Trainer

from ..models import ADDModel


class ADDTrainer(Trainer):
    """Trainer that owns classification loss while ADDModel remains loss-agnostic."""

    def __init__(
        self,
        *args: Any,
        loss_function: nn.Module,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.loss_function = loss_function.to(self.accelerator.device)

    def compute_loss(
        self,
        model: ADDModel,
        inputs: Mapping[str, torch.Tensor],
        return_outputs: bool = False,
        **_: Any,
    ):
        labels = inputs["labels"]
        outputs = model(
            **inputs,
            use_cache=False,
            return_dict=True,
        )
        loss_logits = outputs.loss_logits if outputs.loss_logits is not None else outputs.logits
        loss = self.loss_function(loss_logits.float(), labels)
        if outputs.aux_loss is not None:
            loss = loss + outputs.aux_loss.to(device=loss.device, dtype=loss.dtype)
        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model: ADDModel,
        inputs: dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ):
        """Keep prediction metrics on the classifier logits, not ModelOutput metadata."""

        del ignore_keys
        inputs = self._prepare_inputs(inputs)
        labels = inputs.get("labels")
        with torch.no_grad(), self.compute_loss_context_manager():
            if labels is None:
                outputs = model(**inputs, use_cache=False, return_dict=True)
                loss = None
            else:
                loss, outputs = self.compute_loss(
                    model,
                    dict(inputs),
                    return_outputs=True,
                )
        loss = loss.detach() if loss is not None else None
        if prediction_loss_only:
            return loss, None, None

        logits = outputs.logits.detach()
        labels = labels.detach() if labels is not None else None
        return loss, logits, labels


def compute_metrics(prediction: EvalPrediction) -> dict[str, float]:
    logits = prediction.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    predictions = np.asarray(logits).argmax(axis=-1)
    labels = np.asarray(prediction.label_ids)
    return {"accuracy": float((predictions == labels).mean())}
