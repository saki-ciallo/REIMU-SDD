from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from ..configuration.settings import DataSettings


class AudioClassificationCollator:
    """Stack fixed-length, single-channel waveforms and integer class labels."""

    def __init__(self, settings: DataSettings) -> None:
        self.input_column = settings.input_column
        self.label_column = settings.label_column

    def __call__(
        self,
        features: Sequence[Mapping[str, object]],
    ) -> dict[str, torch.Tensor]:
        if not features:
            raise ValueError("AudioClassificationCollator requires at least one feature.")

        waveforms = []
        for index, feature in enumerate(features):
            waveform = torch.as_tensor(feature[self.input_column], dtype=torch.float32)
            if waveform.ndim != 1:
                raise ValueError(
                    "AudioClassificationCollator expects single-channel waveforms shaped "
                    f"[T]; feature {index} has shape {tuple(waveform.shape)}."
                )
            waveforms.append(waveform)

        lengths = {waveform.numel() for waveform in waveforms}
        if len(lengths) != 1:
            raise ValueError(
                f"Training batches require fixed-length waveforms; got lengths {sorted(lengths)}."
            )
        labels = torch.tensor(
            [int(feature[self.label_column]) for feature in features],
            dtype=torch.long,
        )
        return {"input_values": torch.stack(waveforms), "labels": labels}
