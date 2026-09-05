from __future__ import annotations

from pathlib import Path

from torch import nn

from ..models import ADDModel
from ..models.backbones.architectures.baseline import BaselineArchitecture
from ..models.backbones.architectures.hrm import HRMArchitecture
from ..models.backbones.architectures.looped import LoopedArchitecture
from ..models.backbones.backbone_stack import BackboneStack
from ..models.frontends.ssl import SSLFrontendModel


def parameter_counts(module: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(
        parameter.numel() for parameter in module.parameters() if parameter.requires_grad
    )
    return total, trainable


def _component_table(model: ADDModel) -> list[str]:
    lines = [
        "| Component | Class | Total parameters | Trainable parameters |",
        "|---|---|---:|---:|",
    ]
    for name in ("frontend", "backbone", "pooling", "aasist", "classifier"):
        module = getattr(model, name, None)
        if module is None:
            continue
        total, trainable = parameter_counts(module)
        lines.append(f"| `{name}` | `{module.__class__.__name__}` | {total:,} | {trainable:,} |")
    return lines


def _stack_lines(name: str, stack: BackboneStack) -> list[str]:
    lines = [
        f"### {name}",
        "",
        f"- Post-norm: `{stack.use_post_norm}`",
        f"- Layers: `{len(stack.layers)}`",
        "",
        "| Layer | Block | Mixer | Feed-forward |",
        "|---:|---|---|---|",
    ]
    for index, layer in enumerate(stack.layers):
        lines.append(
            f"| {index} | `{layer.__class__.__name__}` | "
            f"`{layer.attn.__class__.__name__}` | `{layer.mlp.__class__.__name__}` |"
        )
    return lines


def _backbone_lines(model: ADDModel) -> list[str]:
    if model.backbone is None:
        return []
    architecture = model.backbone.architecture
    lines = [
        "## Backbone",
        "",
        f"- Architecture: `{model.config.architecture_type}`",
        f"- Implementation: `{architecture.__class__.__name__}`",
        f"- Hidden size: `{model.backbone.config.hidden_size}`",
        "",
    ]
    if isinstance(architecture, BaselineArchitecture):
        lines.extend(_stack_lines("Baseline stack", architecture.stack))
    elif isinstance(architecture, LoopedArchitecture):
        lines.extend(
            [
                f"- Gradient schedule: `{architecture.gradient_schedule}`",
                "",
                *_stack_lines("Shared looped stack", architecture.stack),
            ]
        )
    elif isinstance(architecture, HRMArchitecture):
        lines.extend(
            [
                f"- H cycles: `{architecture.h_cycles}`",
                f"- L cycles per H: `{architecture.l_cycles}`",
                f"- Gradient schedule: `{architecture.gradient_schedule}`",
                "",
                *_stack_lines("High-level stack", architecture.high_level),
                "",
                *_stack_lines("Low-level stack", architecture.low_level),
            ]
        )
    return lines


def _frontend_lines(model: ADDModel) -> list[str]:
    lines = [
        "## Frontend",
        "",
        f"- Wrapper: `{model.frontend.__class__.__name__}`",
        f"- Config type: `{model.frontend.config.model_type}`",
    ]
    if isinstance(model.frontend, SSLFrontendModel):
        indices = model.frontend.trainable_encoder_layer_indices
        lines.extend(
            [
                f"- SSL model: `{model.frontend.ssl_model.__class__.__name__}`",
                f"- Tuning mode: `{model.frontend.ssl_tuning_mode}`",
                f"- Trainable encoder layers: `{indices}`",
                f"- Output projection: `{model.frontend.output_projection.__class__.__name__}`",
            ]
        )
    return lines


def _head_lines(model: ADDModel) -> list[str]:
    lines: list[str] = []
    if model.pooling is not None:
        lines.extend(
            [
                "## Pooling",
                "",
                f"- Type: `{model.pooling.config.pooling_type}`",
                f"- Implementation: `{model.pooling.pooling.__class__.__name__}`",
                f"- Output size: `{model.pooling.config.output_size}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Classifier",
            "",
            f"- Type: `{model.classifier.config.classifier_type}`",
            f"- Implementation: `{model.classifier.classifier.projection.__class__.__name__}`",
            f"- Number of labels: `{model.classifier.config.num_labels}`",
        ]
    )
    return lines


def _trainable_groups(model: ADDModel) -> list[str]:
    groups: dict[str, int] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group = name.split(".", 1)[0]
        groups[group] = groups.get(group, 0) + parameter.numel()
    lines = [
        "## Trainable Groups",
        "",
        "| Prefix | Parameters |",
        "|---|---:|",
    ]
    lines.extend(f"| `{name}` | {count:,} |" for name, count in sorted(groups.items()))
    return lines


def build_model_report(model: ADDModel) -> str:
    total, trainable = parameter_counts(model)
    lines = [
        "# Model Overview",
        "",
        f"- Pipeline: `{model.config.model_architecture}`",
        f"- Architecture: `{model.config.architecture_type}`",
        f"- Total parameters: `{total:,}`",
        f"- Trainable parameters: `{trainable:,}`",
        "",
        "## Components",
        "",
        *_component_table(model),
        "",
        *_frontend_lines(model),
        "",
        *_backbone_lines(model),
        "",
        *_head_lines(model),
        "",
        *_trainable_groups(model),
        "",
    ]
    return "\n".join(lines)


def write_model_report(model: ADDModel, output_dir: str | Path) -> Path:
    destination = Path(output_dir) / "model_overview.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(build_model_report(model), encoding="utf-8")
    return destination
