from __future__ import annotations

from collections.abc import Mapping
from dataclasses import MISSING, dataclass, fields
from pathlib import Path

from .components.add import ADDConfig
from .experiments import ResolvedExperiment
from .validation import boolean, one_of, positive_float, positive_int, probability

CODE_OPTIMIZE_ROOT = Path(__file__).resolve().parents[2]


def _section[T](
    values: Mapping[str, object],
    section_name: str,
    settings_type: type[T],
) -> T:
    raw = values.get(section_name)
    if not isinstance(raw, Mapping):
        raise TypeError(f"{section_name} must be a mapping.")
    allowed = {field.name for field in fields(settings_type)}
    unknown = set(raw) - allowed
    if unknown:
        names = ", ".join(sorted(str(name) for name in unknown))
        raise ValueError(f"{section_name} contains unknown fields: {names}.")
    required = {
        field.name
        for field in fields(settings_type)
        if field.default is MISSING and field.default_factory is MISSING
    }
    missing = required - set(raw)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"{section_name} is missing required fields: {names}.")
    return settings_type(**raw)


@dataclass(frozen=True, slots=True)
class RunSettings:
    task_name: str
    output_root: str = "outputs"
    seed: int = 42
    gpu_id: int | None = None

    def __post_init__(self) -> None:
        if not self.task_name.strip():
            raise ValueError("run.task_name must not be empty.")
        if not self.output_root.strip():
            raise ValueError("run.output_root must not be empty.")
        positive_int(self.seed, field_name="run.seed")
        if self.gpu_id is not None:
            if isinstance(self.gpu_id, bool) or not isinstance(self.gpu_id, int):
                raise TypeError("run.gpu_id must be an integer or null.")
            if self.gpu_id < 0:
                raise ValueError("run.gpu_id must be non-negative or null.")


@dataclass(frozen=True, slots=True)
class DataSettings:
    dataset_key: str
    dataset_cache_path: str
    train_split: str = "train"
    validation_split: str = "validation"
    input_column: str = "input_values"
    label_column: str = "labels"
    max_train_samples: int | None = None
    max_eval_samples: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "dataset_key",
            "dataset_cache_path",
            "train_split",
            "validation_split",
            "input_column",
            "label_column",
        ):
            if not getattr(self, name).strip():
                raise ValueError(f"data.{name} must not be empty.")
        for name in ("max_train_samples", "max_eval_samples"):
            value = getattr(self, name)
            if value is not None:
                positive_int(value, field_name=f"data.{name}")


@dataclass(frozen=True, slots=True)
class LossSettings:
    loss_type: str = "cross_entropy"
    bonafide_weight: float = 0.8
    spoof_weight: float = 0.2
    focal_gamma: float = 2.0
    label_smoothing: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "loss_type",
            one_of(
                self.loss_type,
                {"cross_entropy", "focal"},
                field_name="loss.loss_type",
            ),
        )
        positive_float(self.bonafide_weight, field_name="loss.bonafide_weight")
        positive_float(self.spoof_weight, field_name="loss.spoof_weight")
        if self.focal_gamma < 0:
            raise ValueError("loss.focal_gamma must be non-negative.")
        if not 0.0 <= self.label_smoothing <= 1.0:
            raise ValueError("loss.label_smoothing must lie in [0, 1].")

    @property
    def class_weights(self) -> tuple[float, float]:
        return self.bonafide_weight, self.spoof_weight


@dataclass(frozen=True, slots=True)
class AugmentationSettings:
    algorithm: int = 0
    apply_to_validation: bool = True
    sampling_rate: int = 16_000
    num_bands: int = 5
    min_frequency: float = 20.0
    max_frequency: float = 8_000.0
    min_bandwidth: float = 100.0
    max_bandwidth: float = 1_000.0
    min_coefficients: int = 10
    max_coefficients: int = 100
    min_gain: float = 0.0
    max_gain: float = 0.0
    min_nonlinear_bias: float = 5.0
    max_nonlinear_bias: float = 20.0
    nonlinearity_order: int = 5
    impulse_percent: float = 10.0
    impulse_gain: float = 2.0
    min_snr: float = 10.0
    max_snr: float = 40.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.algorithm, bool)
            or not isinstance(self.algorithm, int)
            or self.algorithm not in range(9)
        ):
            raise ValueError("augmentation.algorithm must be an integer from 0 through 8.")
        boolean(
            self.apply_to_validation,
            field_name="augmentation.apply_to_validation",
        )
        positive_int(self.sampling_rate, field_name="augmentation.sampling_rate")
        positive_int(self.num_bands, field_name="augmentation.num_bands")
        positive_int(
            self.nonlinearity_order,
            field_name="augmentation.nonlinearity_order",
        )
        if self.max_frequency > self.sampling_rate / 2:
            raise ValueError("augmentation.max_frequency exceeds Nyquist frequency.")
        for name in (
            "frequency",
            "bandwidth",
            "coefficients",
            "nonlinear_bias",
            "snr",
        ):
            minimum = getattr(self, f"min_{name}")
            maximum = getattr(self, f"max_{name}")
            if minimum >= maximum:
                raise ValueError(f"augmentation {name} minimum must be below maximum.")
        if self.min_gain > self.max_gain:
            raise ValueError("augmentation min_gain must not exceed max_gain.")
        if not 0.0 <= self.impulse_percent <= 100.0:
            raise ValueError("augmentation.impulse_percent must lie in [0, 100].")

    @property
    def enabled(self) -> bool:
        return self.algorithm != 0


@dataclass(frozen=True, slots=True)
class TrainingSettings:
    do_train: bool = True
    do_eval: bool = True
    resume_from_checkpoint: str | bool | None = None
    bf16: bool = True
    torch_compile: bool = False
    torch_compile_mode: str = "default"
    optimizer: str = "adamw_torch_fused"
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    num_train_epochs: float = 20.0
    max_steps: int = -1
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    dataloader_num_workers: int = 2
    remove_unused_columns: bool = True
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    logging_strategy: str = "epoch"
    logging_steps: int = 1
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    early_stopping_patience: int = 7
    early_stopping_threshold: float = 0.0
    report_to: list[str] | None = None

    def __post_init__(self) -> None:
        for name in (
            "do_train",
            "do_eval",
            "bf16",
            "torch_compile",
            "load_best_model_at_end",
            "greater_is_better",
            "remove_unused_columns",
        ):
            boolean(getattr(self, name), field_name=f"training.{name}")
        if self.resume_from_checkpoint is not None and not isinstance(
            self.resume_from_checkpoint,
            (str, bool),
        ):
            raise TypeError("training.resume_from_checkpoint must be a path, bool, or null.")
        if isinstance(self.resume_from_checkpoint, str) and not self.resume_from_checkpoint.strip():
            raise ValueError("training.resume_from_checkpoint path must not be empty.")
        if not self.bf16:
            raise ValueError(
                "training.bf16 must be true because FLA attention and grouped MoE "
                "kernels require mixed-precision CUDA execution."
            )
        object.__setattr__(
            self,
            "torch_compile_mode",
            one_of(
                self.torch_compile_mode,
                {"default", "max-autotune-no-cudagraphs"},
                field_name="training.torch_compile_mode",
            ),
        )
        positive_float(self.learning_rate, field_name="training.learning_rate")
        if self.weight_decay < 0:
            raise ValueError("training.weight_decay must be non-negative.")
        probability(self.warmup_ratio, field_name="training.warmup_ratio")
        for name in ("adam_beta1", "adam_beta2"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"training.{name} must be a number.")
            if not 0.0 <= value < 1.0:
                raise ValueError(f"training.{name} must lie in [0, 1).")
        positive_float(self.num_train_epochs, field_name="training.num_train_epochs")
        if self.max_steps == 0 or self.max_steps < -1:
            raise ValueError("training.max_steps must be -1 or a positive integer.")
        for name in (
            "per_device_train_batch_size",
            "per_device_eval_batch_size",
            "gradient_accumulation_steps",
            "logging_steps",
            "save_total_limit",
        ):
            positive_int(getattr(self, name), field_name=f"training.{name}")
        if self.dataloader_num_workers < 0:
            raise ValueError("training.dataloader_num_workers must be non-negative.")
        if self.early_stopping_patience < 0:
            raise ValueError("training.early_stopping_patience must be non-negative.")
        if self.early_stopping_threshold < 0:
            raise ValueError("training.early_stopping_threshold must be non-negative.")
        if self.load_best_model_at_end and not self.do_eval:
            raise ValueError("load_best_model_at_end requires do_eval=True.")
        for name, choices in (
            ("eval_strategy", {"no", "steps", "epoch"}),
            ("save_strategy", {"no", "steps", "epoch"}),
            ("logging_strategy", {"no", "steps", "epoch"}),
        ):
            object.__setattr__(
                self,
                name,
                one_of(
                    getattr(self, name),
                    choices,
                    field_name=f"training.{name}",
                ),
            )
        if self.report_to is not None and (
            not isinstance(self.report_to, list)
            or any(not isinstance(item, str) or not item for item in self.report_to)
        ):
            raise TypeError("training.report_to must be a list of non-empty strings or null.")


@dataclass(frozen=True, slots=True)
class ExperimentSettings:
    run: RunSettings
    data: DataSettings
    model: ADDConfig
    loss: LossSettings
    augmentation: AugmentationSettings
    training: TrainingSettings
    sha256: str

    @property
    def task_name(self) -> str:
        prefix = f"{self.data.dataset_key}_"
        return (
            self.run.task_name
            if self.run.task_name.startswith(prefix)
            else f"{prefix}{self.run.task_name}"
        )

    @property
    def output_dir(self) -> Path:
        output_root = Path(self.run.output_root).expanduser()
        if not output_root.is_absolute():
            output_root = CODE_OPTIMIZE_ROOT / output_root
        return output_root / self.task_name


def build_experiment_settings(resolved: ResolvedExperiment) -> ExperimentSettings:
    allowed_sections = {
        "run",
        "data",
        "model",
        "loss",
        "augmentation",
        "training",
    }
    unknown = set(resolved.values) - allowed_sections
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Experiment contains unknown top-level sections: {names}.")
    return ExperimentSettings(
        run=_section(resolved.values, "run", RunSettings),
        data=_section(resolved.values, "data", DataSettings),
        model=resolved.build_model_config(),
        loss=_section(resolved.values, "loss", LossSettings),
        augmentation=_section(
            resolved.values,
            "augmentation",
            AugmentationSettings,
        ),
        training=_section(resolved.values, "training", TrainingSettings),
        sha256=resolved.sha256,
    )
