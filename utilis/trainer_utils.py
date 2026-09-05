from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import torch
from transformers import Trainer

from addition_loss.focal_loss import FocalLoss
from src.modeling_add import ADDModel


@dataclass
class LossArguments:
    loss_type: str = field(
        default="cross_entropy",
        metadata={"help": "Loss function: cross_entropy or focal."},
    )
    loss_bonafide_weight: float = field(default=0.8)
    loss_spoof_weight: float = field(default=0.2)
    focal_gamma: float = field(default=2.0)
    ce_label_smoothing: float = field(default=0.0)

    def __post_init__(self) -> None:
        if self.loss_type not in {"cross_entropy", "focal"}:
            raise ValueError(
                f"loss_type must be one of cross_entropy or focal, got {self.loss_type!r}."
            )
        if self.loss_bonafide_weight <= 0 or self.loss_spoof_weight <= 0:
            raise ValueError("Loss class weights must be positive.")
        if self.focal_gamma < 0:
            raise ValueError("focal_gamma must be non-negative.")
        if not 0.0 <= self.ce_label_smoothing <= 1.0:
            raise ValueError("ce_label_smoothing must be between 0 and 1.")

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> LossArguments:
        field_names = cls.__dataclass_fields__
        return cls(**{name: values[name] for name in field_names if name in values})

    @property
    def class_weights(self) -> list[float]:
        return [self.loss_bonafide_weight, self.loss_spoof_weight]


def build_loss_fn(loss_args: LossArguments, num_labels: int) -> torch.nn.Module:
    if num_labels != 2:
        raise ValueError("The configured bonafide/spoof class weights require num_labels=2.")

    if loss_args.loss_type == "cross_entropy":
        return torch.nn.CrossEntropyLoss(
            weight=torch.tensor(loss_args.class_weights, dtype=torch.float32),
            label_smoothing=loss_args.ce_label_smoothing,
        )
    return FocalLoss(
        gamma=loss_args.focal_gamma,
        alpha=loss_args.class_weights,
        task_type="multi-class",
        num_classes=num_labels,
    )


class AudioClassificationCollator:
    def __init__(
        self,
        input_values_column_name: str,
        label_column_name: str,
    ) -> None:
        self.input_values_column_name = input_values_column_name
        self.label_column_name = label_column_name

    def _audio_to_tensor(self, example: dict[str, Any]) -> torch.Tensor:
        tensor = torch.as_tensor(example[self.input_values_column_name], dtype=torch.float32)
        if tensor.ndim > 1:
            tensor = tensor.reshape(-1)
        return tensor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_values = torch.stack([self._audio_to_tensor(feature) for feature in features])
        labels = torch.tensor(
            [int(feature[self.label_column_name]) for feature in features],
            dtype=torch.long,
        )
        return {
            "input_values": input_values,
            "labels": labels,
        }


class ADDTrainer(Trainer):
    def __init__(
        self,
        *args,
        loss_fn: torch.nn.Module | None = None,
        # Legacy CUDA RawBoost arguments.  Kept as comments so the former
        # Trainer-side path remains documented but cannot be enabled by accident.
        # cuda_rawboost: torch.nn.Module | None = None,
        # rawboost_apply_to_validation: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        # Legacy CUDA RawBoost state (intentionally disabled):
        # self.cuda_rawboost = cuda_rawboost
        # self.rawboost_apply_to_validation = rawboost_apply_to_validation
        self.loss_fn = (
            loss_fn.to(self.accelerator.device)
            if loss_fn is not None
            else None
        )

    # Legacy CUDA RawBoost helper (intentionally disabled).  The active
    # implementation applies the original CPU RawBoost through Dataset transforms
    # before this Trainer receives a batch.
    # def _apply_cuda_rawboost(
    #     self,
    #     model: ADDModel,
    #     inputs: dict[str, torch.Tensor],
    # ) -> dict[str, torch.Tensor]:
    #     if self.cuda_rawboost is None:
    #         return inputs
    #     if not model.training and not self.rawboost_apply_to_validation:
    #         return inputs
    #     if "input_values" not in inputs:
    #         raise KeyError("CUDA RawBoost requires an input_values tensor.")
    #
    #     augmented_inputs = dict(inputs)
    #     augmented_inputs["input_values"] = self.cuda_rawboost(
    #         inputs["input_values"]
    #     )
    #     return augmented_inputs

    def compute_loss(
        self,
        model: ADDModel,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs: Any,
    ):
        if self.loss_fn is None:
            raise RuntimeError("ADDTrainer requires loss_fn because ADDModel does not compute loss.")

        labels = inputs["labels"]
        # Legacy CUDA RawBoost batch replacement (intentionally disabled):
        # model_inputs = self._apply_cuda_rawboost(model, inputs)
        # outputs = model(**model_inputs, use_cache=False, return_dict=True)
        outputs = model(**inputs, use_cache=False, return_dict=True)
        loss = self.loss_fn(outputs.logits.float(), labels)

        if outputs.aux_loss is not None:
            loss = loss + outputs.aux_loss.to(device=loss.device, dtype=loss.dtype)

        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model: ADDModel,
        inputs: dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ):
        inputs = self._prepare_inputs(inputs)
        labels = inputs.get("labels")
        with torch.no_grad(), self.compute_loss_context_manager():
            loss, outputs = self.compute_loss(model, dict(inputs), return_outputs=True)
        loss = loss.detach()
        if prediction_loss_only:
            return loss, None, None
        logits = outputs.logits.detach()
        labels = labels.detach() if labels is not None else None
        return loss, logits, labels


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    accuracy = (predictions == labels).mean().item()
    return {"accuracy": accuracy}


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits


__all__ = [
    "ADDTrainer",
    "AudioClassificationCollator",
    "LossArguments",
    "build_loss_fn",
    "compute_metrics",
    "preprocess_logits_for_metrics",
]
