from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class LinearAudioFrontend(nn.Module):
    """Slice raw audio into non-overlapping frames and project each frame."""

    def __init__(
        self,
        frame_size: int,
        output_size: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if frame_size <= 0:
            raise ValueError("frame_size must be positive.")
        if output_size <= 0:
            raise ValueError("output_size must be positive.")
        self.frame_size = frame_size
        self.output_size = output_size
        self.proj = nn.Linear(frame_size, output_size, bias=bias)

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        if input_values.ndim != 2:
            raise ValueError("input_values must have shape [batch_size, num_samples].")
        num_frames = input_values.shape[-1] // self.frame_size
        if num_frames <= 0:
            raise ValueError(
                f"input_values must contain at least one full frame of {self.frame_size} samples."
            )
        usable_samples = num_frames * self.frame_size
        hidden_state = input_values[:, :usable_samples].reshape(
            input_values.shape[0],
            num_frames,
            self.frame_size,
        ) # 取整，不足的frame丢弃
        return self.proj(hidden_state)


@dataclass
class StreamingSincState:
    """
    SincConv 流式状态
    SincConv 每个输出 frame 需要一个长度为 kernel_size 的原始波形窗口。如果当前 chunk 不足以构成完整窗口，就存放在 raw_buffer 中，等待后续 chunk 新音频到来时拼接处理。
    """
    raw_buffer: Optional[torch.Tensor] = None # 用于存储未处理的原始音频片段
    produced_frames: int = 0 # 记录输出的 frame 数量
    raw_start: int = 0 # 记录新的起始位置，raw_buffer[:, 0]，也是 sample index


@dataclass
class StreamingMixedCNNState:
    """
    保留每一层的 conv state
    每个 layer_state 是该层输入的最后 kernel_size - 1 个 token。
    下次卷积使用过去的输入上下文。
    """
    layer_states: list[Optional[torch.Tensor]] = field(default_factory=list)


class SincConv_fast(nn.Module):
    """Vectorized SincNet convolution based on the official ``SincConv_fast``.

    Trainable parameters:
    - ``low_hz_``: lower cutoff for each filter in Hz, shape ``[out_channels, 1]``.
    - ``band_hz_``: bandwidth for each filter in Hz, shape ``[out_channels, 1]``.

    The actual convolution kernels are generated vectorially at every forward
    pass. There are no independent dense Conv1d weights.
    """

    def __init__(
        self,
        out_channels: int,
        kernel_size: int, # window_size, 16000*0.025=400个采样点，每个卷积核覆盖 25ms 的音频
        sample_rate: int = 16_000,
        in_channels: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
        groups: int = 1,
        min_low_hz: float = 50.0,
        min_band_hz: float = 50.0,
    ) -> None:
        super().__init__()
        if in_channels != 1:
            raise ValueError(f"SincConv only supports one input channel, got {in_channels}.")
        if bias:
            raise ValueError("SincConv does not support bias.")
        if groups > 1:
            raise ValueError("SincConv does not support groups.")
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive.")
        if stride <= 0:
            raise ValueError("stride must be positive.")
        if out_channels <= 0:
            raise ValueError("out_channels must be positive.")

        self.out_channels = out_channels
        self.requested_kernel_size = kernel_size
        self.kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size # 中心点对齐，奇数
        self.sample_rate = sample_rate
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        low_hz, band_hz = self._mel_initialization(out_channels, sample_rate)
        self.low_hz_ = nn.Parameter(low_hz.view(-1, 1))
        self.band_hz_ = nn.Parameter(band_hz.view(-1, 1))

    @staticmethod
    def to_mel(hz: torch.Tensor) -> torch.Tensor:
        return 2595.0 * torch.log10(torch.ones((), device=hz.device, dtype=hz.dtype) + hz / 700.0)

    @staticmethod
    def to_hz(mel: torch.Tensor) -> torch.Tensor:
        return 700.0 * (torch.pow(torch.tensor(10.0, device=mel.device, dtype=mel.dtype), mel / 2595.0) - 1.0)

    def _mel_initialization(
        self,
        out_channels: int,
        sample_rate: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        low_hz = torch.tensor(30.0, dtype=torch.float32)
        high_hz = torch.tensor(
            sample_rate / 2.0 - (self.min_low_hz + self.min_band_hz),
            dtype=torch.float32,
        )
        mel = torch.linspace(self.to_mel(low_hz), self.to_mel(high_hz), out_channels + 1)
        hz = self.to_hz(mel)
        return hz[:-1], torch.diff(hz)

    def filters(self) -> torch.Tensor:
        # Build the deterministic basis at runtime. Non-persistent buffers created
        # on HF's meta device become uninitialized after from_pretrained().
        with torch.autocast(device_type=self.low_hz_.device.type, enabled=False):
            low = self.min_low_hz + torch.abs(self.low_hz_.float())
            high = torch.clamp(
                low + self.min_band_hz + torch.abs(self.band_hz_.float()),
                self.min_low_hz,
                self.sample_rate / 2.0,
            )
            band = (high - low)[:, 0]

            half_kernel = self.kernel_size // 2
            n_lin = torch.arange(
                half_kernel,
                device=low.device,
                dtype=low.dtype,
            )
            window = 0.54 - 0.46 * torch.cos(
                2.0 * torch.pi * n_lin / self.kernel_size
            )
            negative_time = (
                2.0
                * torch.pi
                * torch.arange(
                    -half_kernel,
                    0,
                    device=low.device,
                    dtype=low.dtype,
                ).view(1, -1)
                / self.sample_rate
            )
            f_times_t_low = torch.matmul(low, negative_time)
            f_times_t_high = torch.matmul(high, negative_time)

            denominator = negative_time / 2.0
            band_pass_left = (
                (torch.sin(f_times_t_high) - torch.sin(f_times_t_low))
                / denominator
            ) * window
            band_pass_center = 2.0 * band.view(-1, 1)
            band_pass_right = torch.flip(band_pass_left, dims=[1])
            band_pass = torch.cat(
                [band_pass_left, band_pass_center, band_pass_right],
                dim=1,
            )
            band_pass = band_pass / (2.0 * band[:, None])
            return band_pass.view(self.out_channels, 1, self.kernel_size)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        if waveforms.ndim == 2:
            waveforms = waveforms.unsqueeze(1)
        if waveforms.ndim != 3 or waveforms.shape[1] != 1:
            raise ValueError("waveforms must have shape [batch_size, num_samples] or [batch_size, 1, num_samples].")
        return F.conv1d(
            waveforms,
            self.filters(),
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=None,
            groups=1,
        ) # shape [B, out_channels, T_sinc]

    def reset_streaming_state(self) -> StreamingSincState:
        """
        创建新的 streaming state。新音频或 session 的时候用。
        """
        return StreamingSincState()

    def forward_streaming(
        self,
        new_waveforms: torch.Tensor,
        state: Optional[StreamingSincState] = None,
    ) -> Tuple[Optional[torch.Tensor], StreamingSincState]:
        if self.padding != 0:
            raise NotImplementedError("Streaming SincConv currently requires padding=0.")
        if new_waveforms.ndim == 3 and new_waveforms.shape[1] == 1:
            new_waveforms = new_waveforms.squeeze(1)
        if new_waveforms.ndim != 2:
            raise ValueError("new_waveforms must have shape [batch_size, new_samples] or [batch_size, 1, new_samples].")

        new_waveforms = new_waveforms.to(device=self.low_hz_.device, dtype=self.low_hz_.dtype) # 混合精度控制，fp32和bf16都行

        state = self.reset_streaming_state() if state is None else state
        if state.raw_buffer is None: # 如果是新的，保存当前的原始音频
            state.raw_buffer = new_waveforms # [0, 16000]
            state.raw_start = 0
        else:
            state.raw_buffer = state.raw_buffer.to(device=new_waveforms.device, dtype=new_waveforms.dtype)
            if state.raw_buffer.ndim != 2:
                raise ValueError("state.raw_buffer must have shape [batch_size, buffered_samples].")
            if state.raw_buffer.shape[0] != new_waveforms.shape[0]:
                raise ValueError("new_waveforms batch size must match state.raw_buffer batch size.")
            state.raw_buffer = torch.cat([state.raw_buffer, new_waveforms], dim=-1) # 有就拼接
            # 假如初始 1 秒，16000，第二轮 0.5 秒传入 8000.
            # steraming 后 raw_buffer [15680, 16000]，拼接得到 [15680, 24000], 320 + 8000 = 8320 个新采样点

        # 计算当前长度能产生多少 sinc frames
        effective_kernel_size = self.dilation * (self.kernel_size - 1) + 1 # 卷积核感受野大小, 401个采样点
        total_available_end = state.raw_start + state.raw_buffer.shape[-1] # 0 + 16000 = 16000
        next_frame = state.produced_frames # 下一帧的起始位置，初始 0，后 0.5 秒是 49
        next_start = next_frame * self.stride # 0 * 320 = 0；16000/0.02=320个采样点对应 20ms 的步长，滑动窗口
        max_frames_total = max(0, (total_available_end - effective_kernel_size) // self.stride + 1)
        # example: (16000 - 401) // 320 + 1 = 49 frames, [0,49]
        # (24000 - 401) // 320 + 1 = 74，现在全局可以产生 74 个frames，但之前已经输出 0...48，现在是 49...73 共计 25 个新frames

        if max_frames_total <= next_frame: # 至少需要一个完整的 kernel 大小的窗口才能产生新的一帧
            state.raw_buffer = state.raw_buffer.detach() # 保存当前的 raw_buffer，等待后续音频到来时继续拼接处理
            return None, state

        local_start = next_start - state.raw_start # 0 - 0 = 0，下一帧的起始位置在 raw_buffer 中的索引
        if local_start < 0: # there's a problem with my AI —— neuro-sama
            raise ValueError("StreamingSincState is inconsistent: next frame starts before raw_buffer.")

        # 从未输出 frame 的起点截取 raw audio，新得到的 sinc features 全是新的 frames
        last_frame = max_frames_total - 1 # 49 - 1 = 48 最后一个新 frame
        needed_end = last_frame * self.stride + effective_kernel_size
        # 48 * 320 + 401 = 15761
        # 73 * 320 + 401 = 23761
        local_end = needed_end - state.raw_start
        # 15761 - 0 = 15761      覆盖 0...48 共 49 个 frames 的原始音频窗口
        # 23761 - 15680 = 8081   覆盖 49...73 共 25 个 frames 的原始音频窗口，前面 320 个采样点是 overlap，后面 7761 个采样点是新音频
        waveforms = state.raw_buffer[:, local_start:local_end] # 实际传给 SincConv 的原始音频窗口 [B, 0:15761]
        sinc_features = self.forward(waveforms) # 49 frames, shape [B, out_channels, 49]
        # 第二次 0.5 秒传入的是 raw_buffer[:, 0:8081]，对应 [15680, 23761]
        
        state.produced_frames = max_frames_total # 49
        keep_from_global = state.produced_frames * self.stride
        # 49 * 320 = 15680，下一个还没输出的 frame 是 49，这就是起点
        # 74 * 32 = 23680
        keep_from_local = keep_from_global - state.raw_start # 15680 - 0 = 15680; 23680 - 15680 = 8000
        state.raw_buffer = state.raw_buffer[:, keep_from_local:].detach() # [:, 15680:16000]，剩下的 320 个采样点，等待下次新音频到来时继续拼接处理，为 frame 跨边界保留的 overlap
        # 后 0.5 秒 raw_buffer[:, 8000:]，对应原始的 [23680, 24000] 始终会被保留给下一次
        state.raw_start = keep_from_global # 15680

        return sinc_features, state


class CausalCNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_size: int,
        bias: bool,
        rms_norm_eps: float,
        apply_activation: bool = True,
    ) -> None:
        """
        Pad -> Conv1d -> RMSNorm -> if(SiLU)
        """
        super().__init__()
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive.")

        self.conv = nn.Conv1d(
            input_dim,
            output_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.norm = nn.RMSNorm(output_dim, eps=rms_norm_eps)
        self.activation = nn.SiLU()
        self.kernel_size = kernel_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.apply_activation = apply_activation

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = F.pad(hidden_states, (self.kernel_size - 1, 0))
        hidden_states = self.conv(hidden_states)
        hidden_states = self.norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        if self.apply_activation:
            hidden_states = self.activation(hidden_states)
        return hidden_states

    def forward_streaming(
        self,
        hidden_states: torch.Tensor,
        conv_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if hidden_states.ndim != 3 or hidden_states.shape[1] != self.input_dim:
            raise ValueError(
                f"hidden_states must have shape [batch_size, {self.input_dim}, time]."
            )

        state_len = self.kernel_size - 1
        if state_len == 0:
            padded = hidden_states
            new_state = None
        else:
            if conv_state is None:
                padded = F.pad(hidden_states, (state_len, 0)) # 与标准 forward 保持一致
            else:
                # steaming 时新进来的 chunk 不能再到前面 pad 0了，需要使用保存的状态在末尾进行拼接
                expected_shape = (hidden_states.shape[0], self.input_dim, state_len)
                if tuple(conv_state.shape) != expected_shape:
                    raise ValueError(
                        f"conv_state must have shape {expected_shape}, got {tuple(conv_state.shape)}."
                    )
                conv_state = conv_state.to(device=hidden_states.device, dtype=hidden_states.dtype)
                padded = torch.cat([conv_state, hidden_states], dim=-1)
            new_state = padded[:, :, -state_len:].detach() # 取最后 state_len 个 token 保存

        hidden_states = self.conv(padded)
        hidden_states = self.norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        if self.apply_activation:
            hidden_states = self.activation(hidden_states)
        return hidden_states, new_state


class MixedCNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        kernels: Sequence[int],
        channels: Sequence[int],
        bias: bool,
        apply_final_cnn_activation: bool,
        use_final_sinc_residual: bool,
        final_dropout: float,
        rms_norm_eps: float,
    ) -> None:
        super().__init__()
        if not kernels:
            raise ValueError("cnn_kernels must not be empty.")
        if len(kernels) != len(channels):
            raise ValueError("cnn_kernels length must match cnn_channels length.")
        if any(kernel_size <= 0 for kernel_size in kernels):
            raise ValueError("all cnn_kernels values must be positive.")
        if any(channel <= 0 for channel in channels):
            raise ValueError("all cnn_channels values must be positive.")

        layers = []
        current_dim = input_dim
        for layer_idx, (kernel_size, next_dim) in enumerate(zip(kernels, channels)):
            is_final_layer = layer_idx == len(kernels) - 1
            layers.append(
                CausalCNN(
                    input_dim=current_dim,
                    output_dim=next_dim,
                    kernel_size=kernel_size,
                    bias=bias,
                    rms_norm_eps=rms_norm_eps,
                    apply_activation=apply_final_cnn_activation if is_final_layer else True, # 最后一层不需要激活函数
                )
            )
            current_dim = next_dim
        self.layers = nn.ModuleList(layers)
        self.output_dim = current_dim
        self.kernels = list(kernels)
        self.channels = list(channels)
        self.use_final_sinc_residual = use_final_sinc_residual
        self.final_residual_proj = ( # 保留实现，但一般没必要
            nn.Conv1d(input_dim, self.output_dim, kernel_size=1, stride=1, padding=0, bias=bias)
            if use_final_sinc_residual
            else None
        )
        self.final_activation = nn.SiLU()
        self.final_dropout = nn.Dropout(final_dropout) if final_dropout > 0.0 else nn.Identity()

    def init_streaming_state(self) -> StreamingMixedCNNState:
        return StreamingMixedCNNState(layer_states=[None for _ in self.layers])

    def reset_streaming_state(self) -> StreamingMixedCNNState:
        return self.init_streaming_state()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        sinc_features = hidden_states
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        if self.use_final_sinc_residual:
            hidden_states = hidden_states + self.final_residual_proj(sinc_features)
        hidden_states = self.final_activation(hidden_states)
        return self.final_dropout(hidden_states)

    def forward_streaming(
        self,
        hidden_states: torch.Tensor,
        state: Optional[StreamingMixedCNNState] = None,
    ) -> Tuple[torch.Tensor, StreamingMixedCNNState]:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch_size, channels, time].")
        if hidden_states.shape[1] != self.layers[0].input_dim:
            raise ValueError(
                f"hidden_states channel dimension must be {self.layers[0].input_dim}, "
                f"got {hidden_states.shape[1]}."
            )

        state = self.init_streaming_state() if state is None or not state.layer_states else state
        if len(state.layer_states) != len(self.layers):
            raise ValueError(
                f"state.layer_states must contain {len(self.layers)} entries, "
                f"got {len(state.layer_states)}."
            )

        sinc_features = hidden_states
        new_layer_states = []
        for layer, layer_state in zip(self.layers, state.layer_states):
            hidden_states, layer_state = layer.forward_streaming(hidden_states, layer_state)
            new_layer_states.append(layer_state)

        if self.use_final_sinc_residual:
            hidden_states = hidden_states + self.final_residual_proj(sinc_features)
        hidden_states = self.final_activation(hidden_states)
        hidden_states = self.final_dropout(hidden_states)
        return hidden_states, StreamingMixedCNNState(layer_states=new_layer_states)
