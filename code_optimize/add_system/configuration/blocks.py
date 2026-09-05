from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class PipelineType(StrEnum):
    BACKBONE_POOLING = "backbone_pooling"
    SSL_AASIST = "ssl_aasist"


class ArchitectureType(StrEnum):
    BASELINE = "baseline"
    LOOPED = "looped"
    HRM = "hrm"
    HETEROGENEOUS_HRM = "heterogeneous_hrm"


class MixerType(StrEnum):
    ATTENTION = "attention"
    RAVEN = "raven"
    GDN2 = "gdn2"
    MAMBA3 = "mamba3"


class FeedForwardType(StrEnum):
    MLP = "mlp"
    MOE = "moe"


type RawBlockSpec = Mapping[str, object] | Sequence[object]
type RawBlockCollection = Mapping[object, RawBlockSpec] | Sequence[RawBlockSpec]


def _positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer, got {type(value).__name__}.")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive, got {value}.")
    return value


def _enum_value[T: StrEnum](
    enum_type: type[T],
    value: object,
    *,
    field_name: str,
) -> T:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string, got {type(value).__name__}.")
    try:
        return enum_type(value.strip().lower())
    except ValueError as exc:
        supported = ", ".join(member.value for member in enum_type)
        raise ValueError(f"{field_name} must be one of {supported}, got {value!r}.") from exc


@dataclass(frozen=True, slots=True)
class BlockSpec:
    """One repeated mixer/feed-forward block group."""

    mixer: MixerType
    feed_forward: FeedForwardType
    num_layers: int = 1

    def __post_init__(self) -> None:
        _positive_int(self.num_layers, field_name="num_layers")

    @classmethod
    def from_raw(cls, raw: RawBlockSpec, *, location: str) -> BlockSpec:
        if isinstance(raw, Mapping):
            unknown = set(raw) - {"attn", "mlp", "num_layers"}
            if unknown:
                names = ", ".join(sorted(str(name) for name in unknown))
                raise ValueError(f"{location} contains unknown fields: {names}.")
            if "attn" not in raw or "mlp" not in raw:
                raise ValueError(f"{location} must define both 'attn' and 'mlp'.")
            mixer = raw["attn"]
            feed_forward = raw["mlp"]
            num_layers = raw.get("num_layers", 1)
        elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            if len(raw) != 3:
                raise ValueError(
                    f"{location} must contain attn, mlp, and num_layers; got {len(raw)} items."
                )
            mixer, feed_forward, num_layers = raw
        else:
            raise TypeError(
                f"{location} must be a mapping or three-item sequence, got {type(raw).__name__}."
            )

        return cls(
            mixer=_enum_value(MixerType, mixer, field_name=f"{location}.attn"),
            feed_forward=_enum_value(
                FeedForwardType,
                feed_forward,
                field_name=f"{location}.mlp",
            ),
            num_layers=_positive_int(
                num_layers,
                field_name=f"{location}.num_layers",
            ),
        )

    def to_legacy_dict(self) -> dict[str, object]:
        """Return the JSON-compatible shape used by existing checkpoints."""

        return {
            "attn": self.mixer.value,
            "mlp": self.feed_forward.value,
            "num_layers": self.num_layers,
        }


def _sortable_block_key(value: object) -> tuple[int, int | str]:
    if isinstance(value, bool):
        return (1, str(value))
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        return (1, str(value))


def normalize_block_specs(raw: RawBlockCollection) -> tuple[BlockSpec, ...]:
    """Normalize mapping/list syntax once, before model construction."""

    if isinstance(raw, Mapping):
        entries = sorted(raw.items(), key=lambda item: _sortable_block_key(item[0]))
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        entries = list(enumerate(raw))
    else:
        raise TypeError(f"blocks must be a mapping or sequence, got {type(raw).__name__}.")

    specs = tuple(BlockSpec.from_raw(spec, location=f"blocks[{key!r}]") for key, spec in entries)
    if not specs:
        raise ValueError("blocks must contain at least one block specification.")
    return specs


def expand_block_specs(specs: Sequence[BlockSpec]) -> tuple[BlockSpec, ...]:
    """Expand grouped specs to one immutable specification per layer."""

    return tuple(
        BlockSpec(spec.mixer, spec.feed_forward) for spec in specs for _ in range(spec.num_layers)
    )


def legacy_block_dicts(specs: Sequence[BlockSpec]) -> list[dict[str, Any]]:
    return [spec.to_legacy_dict() for spec in specs]
