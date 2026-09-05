from __future__ import annotations

import math
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Protocol, TypedDict

import numpy as np
import torch
from datasets import ClassLabel, Value, Audio, load_dataset
from datasets import DatasetDict

from torchcodec.decoders import AudioDecoder


class AudioSamples(Protocol):
    data: torch.Tensor
    sample_rate: int


class AudioDecoderLike(Protocol):
    def get_all_samples(self) -> AudioSamples: ...


class AudioPathEntry(TypedDict, total=False):
    bytes: bytes | None
    path: str | None


class AudioBatch(TypedDict, total=False):
    audio: list[AudioDecoderLike | AudioPathEntry]
    file_name: list[str]
    path: list[str]
    utterance_id: list[str]
    input_values: list[torch.Tensor]


@dataclass
class AudioTransformConfig:
    """音频预处理配置。

    默认路径保持 Hugging Face Audio/TorchCodec 解码；只有
    exception_audio_files 中列出的样本在 TorchCodec 失败后才回退到 ffmpeg。
    """

    sampling_rate: int = 16_000
    target_seconds: float = 4.0
    trim_select: str = "fixed" # fixed 固定长度， various 小于目标长度的不处理
    label_column: str = "labels" # 数据集配置文件一般会处理为 labels 列，不需要改变
    label_names: List[str] = field(default_factory=lambda: ["bonafide", "spoof"]) # 可能存在 bonafide -> bona-fide
    use_dataset_cache: bool = True # 需要检查硬盘空间是否足够
    exception_audio_files: List[str] = field(default_factory=list) # TorchCodec 解码失败时使用 ffmpeg 回退

    def __post_init__(self) -> None:
        if self.sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive.")
        if self.target_seconds <= 0:
            raise ValueError("target_seconds must be positive.")
        if self.trim_select not in {"fixed", "various"}:
            raise ValueError("trim_select must be 'fixed' or 'various'.")

    @property
    def target_length(self) -> int:
        return int(round(self.sampling_rate * self.target_seconds))


def pad_or_trim_waveform(waveform: torch.Tensor, target_length: int, trim_select: str = "fixed") -> torch.Tensor:
    """按目标长度裁剪或处理短音频。

    fixed: 短音频通过重复自身补齐，避免补零引入静音片段。
    various: 只裁剪超长音频，短音频保持原始长度。
    """

    num_samples = waveform.shape[-1] # torch.Size([1, samlpes])

    if trim_select not in {"fixed", "various"}:
        raise ValueError("trim_select must be 'fixed' or 'various'.")

    if num_samples == target_length:
        return waveform
    if num_samples > target_length:
        return waveform[..., :target_length]
    if num_samples == 0:
        raise ValueError("Cannot pad an empty waveform.")
    if trim_select == "various":
        return waveform #

    repeat_count = int(math.ceil(target_length / num_samples))
    return waveform.repeat(1, repeat_count)[..., :target_length]


def _audio_lookup_keys(value: str) -> set[str]:
    """为同一个音频标识生成可匹配的 key。

    允许 exception_audio_files 传入完整路径、文件名或 utterance_id。
    """

    path = Path(value)
    keys = {value, path.as_posix(), path.name, path.stem}
    if path.suffix:
        keys.add(path.with_suffix("").as_posix())
    else:
        keys.add(f"{value}.flac")
        keys.add(f"{path.name}.flac")
    return keys


def _resolve_exception_audio_path(audio_file: str, data_dir: str | Path | None = None) -> Path:
    """把例外音频标识解析成可读路径。

    当传入的是 utterance_id 时，会优先在 data_dir/test 等常见 split 目录下查找 .flac。
    找不到时保留原值，后续只有真正触发 fallback 时才会暴露文件不存在问题。
    """

    path = Path(audio_file)
    candidates = [path]
    if not path.suffix:
        candidates.append(path.with_suffix(".flac"))

    if data_dir is not None and not path.is_absolute():
        base = Path(data_dir)
        candidates.extend(
            [
                base / path,
                base / path.name,
                base / "test" / path.name,
                base / "train" / path.name,
                base / "validation" / path.name,
                base / "dev" / path.name,
            ]
        )
        if not path.suffix:
            filename = f"{path.name}.flac"
            candidates.extend(
                [
                    base / filename,
                    base / "test" / filename,
                    base / "train" / filename,
                    base / "validation" / filename,
                    base / "dev" / filename,
                ]
            )

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def _build_exception_audio_paths(
    exception_audio_files: list[str],
    data_dir: str | Path | None = None,
) -> dict[str, Path]:
    """建立例外音频索引，用于快速判断当前样本是否允许 fallback。"""

    exception_paths: dict[str, Path] = {}
    for audio_file in exception_audio_files:
        path = _resolve_exception_audio_path(audio_file, data_dir)
        for key in _audio_lookup_keys(audio_file) | _audio_lookup_keys(path.as_posix()):
            exception_paths[key] = path
    return exception_paths


def _find_exception_audio_path(
    batch: AudioBatch,
    index: int,
    exception_audio_paths: dict[str, Path],
) -> Path | None:
    """根据当前 batch 的 path/file_name/utterance_id 定位例外音频路径。"""

    audio = batch["audio"][index]
    if isinstance(audio, dict):
        path = audio.get("path")
        if path is not None:
            for key in _audio_lookup_keys(str(path)):
                if key in exception_audio_paths:
                    return exception_audio_paths[key]

    for column in ("file_name", "path", "utterance_id"):
        values = batch.get(column)
        if values is None:
            continue
        value = str(values[index])
        for key in _audio_lookup_keys(value):
            if key in exception_audio_paths:
                return exception_audio_paths[key]

    return None


def _load_audio_with_ffmpeg(path: Path, sampling_rate: int) -> torch.Tensor:
    """使用系统 ffmpeg 解码并重采样，返回 [1, samples] 的 float32 Tensor。"""

    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(path),
        "-ac",
        "1",
        "-ar",
        str(sampling_rate),
        "-f",
        "f32le",
        "pipe:1",
    ]
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stderr = process.stderr.decode("utf-8", errors="replace").strip()
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to decode {path}: {stderr}")

    waveform = np.frombuffer(process.stdout, dtype=np.float32).copy()
    if waveform.size == 0:
        raise RuntimeError(f"ffmpeg decoded no samples from {path}: {stderr}")
    return torch.from_numpy(waveform).unsqueeze(0)


def _decode_audio(
    batch: AudioBatch,
    index: int,
    config: AudioTransformConfig,
    exception_audio_paths: dict[str, Path],
) -> torch.Tensor:
    """优先用 Hugging Face/TorchCodec 解码，命中例外且失败时才回退 ffmpeg。"""

    audio = batch["audio"][index]

    try:
        if isinstance(audio, dict):
            path = audio.get("path")
            if path is None:
                raise ValueError("Decoded-disabled audio entry does not contain a path.")

            samples = AudioDecoder(str(path), sample_rate=config.sampling_rate).get_all_samples()
        else:
            samples = audio.get_all_samples()
        return samples.data
    except Exception:
        # 非例外音频保持原始错误，避免把真实数据问题悄悄吞掉。
        fallback_path = _find_exception_audio_path(batch, index, exception_audio_paths)
        if fallback_path is None:
            raise
        return _load_audio_with_ffmpeg(fallback_path, config.sampling_rate)


def audio_transform(
    config: AudioTransformConfig,
    data_dir: str | Path | None = None,
) -> Callable[[AudioBatch], AudioBatch]:
    """构建给 Dataset.map / Dataset.with_transform 使用的 batch transform。"""

    exception_audio_paths = _build_exception_audio_paths(config.exception_audio_files, data_dir)

    def transform(batch: AudioBatch) -> AudioBatch:
        waveforms = []

        for index in range(len(batch["audio"])):
            waveform = _decode_audio(batch, index, config, exception_audio_paths)

            waveform = pad_or_trim_waveform(waveform, config.target_length, config.trim_select)
            waveforms.append(waveform.squeeze(0)) # 去掉通道维度，变成一维张量。
            # 因为 AudioDecoder.get_all_samples() -> AudioSamples 返回 (num_channels, num_samples)
        if not config.use_dataset_cache:
            batch.pop("audio", None) # 会影响 DataLoader，audio 是音频类不支持

        batch["input_values"] = waveforms # 固定列名称，符合 hugging face 的音频输入名称习惯
        return batch

    return transform


def load_audio_dataset(
    data_dir: str,
    config: AudioTransformConfig,
) -> DatasetDict:
    """加载 audiofolder 数据集并生成 ADD 训练/测试使用的 input_values。

    输出流程：
    1. 使用 Hugging Face audiofolder 读取音频路径和 metadata。
    2. 将 audio 列 cast 为指定采样率的 Audio 特征。
    3. 将标签列转换为 ClassLabel。
    4. 通过 audio_transform 解码、裁剪/补齐，并生成 input_values。
    """

    ds = load_dataset("audiofolder", data_dir=data_dir, drop_labels=True) # 处理音频路径得到 audio 字段，保存的是 AudioDecoder 类
    ds = ds.cast_column("audio", Audio(sampling_rate=config.sampling_rate))
    # ds = ds.rename_column("key", "label") # 暂不使用
    # ds = ds.class_encode_column(config.label_column) # 按首个字符顺序编码
    ds = ds.cast_column(config.label_column, Value("string")) # large_string 需要转换 string 才能正常工作
    ds = ds.cast_column(config.label_column, ClassLabel(names=config.label_names)) # 将字段按位置转成 0, 1
    
    if config.use_dataset_cache:
        ds = ds.map(audio_transform(config, data_dir=data_dir), batched=True, remove_columns=["audio"])
        ds = ds.with_format(
            "torch",
            columns=["input_values", config.label_column],
            output_all_columns=True,
        ) # 可以直接给 DataLoader 使用
    else:
        ds = ds.with_transform(audio_transform(config, data_dir=data_dir))
    if not isinstance(ds, DatasetDict):
        raise TypeError(f"Expected DatasetDict, got {type(ds)}")
    
    return ds


__all__ = [
    "AudioTransformConfig",
    "load_audio_dataset",
    "pad_or_trim_waveform",
]
