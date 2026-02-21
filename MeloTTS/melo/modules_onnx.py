# melo/models/modules_onnx.py
# ============================================================
# ONNX 전용 모듈 구현 (forward-only, reverse/logdet/randn 제거)
#
# 목적:
# - 기존 melo/models/modules.py 는 건드리지 않는다.
# - ONNX export / ONNX Runtime 추론을 위해 "그래프가 고정되는" 경로만 제공한다.
#
# 핵심 원칙(ONNX 안정성):
# 1) reverse=True / 역변환 제거 (flow는 forward-only)
# 2) logdet 반환 제거 (추론에 불필요)
# 3) torch.randn / randn_like / Python list 조작 최소화
# 4) 가능하면 텐서 연산만으로 구성
#
# 주의:
# - 이 파일은 "학습용"이 아니다. 오직 ONNX 추론 그래프를 만들기 위한 전용 구현이다.
# - SynthesizerTrnONNX 같은 래퍼에서 이 모듈들을 사용하도록 분기해야 한다.
# ============================================================

from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from torch.nn import Conv1d
from torch.nn.utils import weight_norm, remove_weight_norm

# ------------------------------------------------------------
# 내부 의존 모듈 (기존 코드 재사용)
# - commons: fused_add_tanh_sigmoid_multiply, init_weights, get_padding 등
# - attentions.Encoder: TransformerCouplingLayer에 사용 가능(원본과 동일)
# - transforms_onnx: piecewise_rational_quadratic_transform (ONNX-safe 버전)
# ------------------------------------------------------------
from . import commons
from .commons import init_weights, get_padding
from .attentions import Encoder
from .transforms_onnx import piecewise_rational_quadratic_transform


LRELU_SLOPE = 0.1


# ============================================================
# Basic layers
# ============================================================

class LayerNorm(nn.Module):
    """
    Conv1d 계열 텐서 [B, C, T]를 대상으로 하는 LayerNorm.
    - ONNX에서는 F.layer_norm이 안정적으로 변환되는 편이라 그대로 사용.
    """
    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [배치(B), 채널(C), 길이(T)]
        x = x.transpose(1, -1)  # [B, T, C]
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        x = x.transpose(1, -1)  # [B, C, T]
        return x


class ConvReluNorm(nn.Module):
    """
    Conv1d + LayerNorm + ReLU + Dropout 블록.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        kernel_size: int,
        n_layers: int,
        p_dropout: float,
    ) -> None:
        super().__init__()
        assert n_layers > 1, "n_layers는 최소 2 이상이어야 함"

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout

        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        # 첫 레이어
        self.conv_layers.append(
            nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=kernel_size // 2)
        )
        self.norm_layers.append(LayerNorm(hidden_channels))

        # 이후 레이어
        for _ in range(n_layers - 1):
            self.conv_layers.append(
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2)
            )
            self.norm_layers.append(LayerNorm(hidden_channels))

        self.relu_drop = nn.Sequential(nn.ReLU(), nn.Dropout(p_dropout))

        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
        # residual 시작 시 안정화를 위해 zero init (원본 관례)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T], x_mask: [B, 1, T]
        x_org = x
        for i in range(self.n_layers):
            x = self.conv_layers[i](x * x_mask)
            x = self.norm_layers[i](x)
            x = self.relu_drop(x)
        x = x_org + self.proj(x)
        return x * x_mask


class DDSConv(nn.Module):
    """
    Dilated Depth-Separable Convolution (원본과 동일 개념)
    - ONNX에서 groups convolution / dilation은 일반적으로 지원됨(opset 17 기준).
    """
    def __init__(self, channels: int, kernel_size: int, n_layers: int, p_dropout: float = 0.0) -> None:
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout

        self.drop = nn.Dropout(p_dropout)
        self.convs_sep = nn.ModuleList()
        self.convs_1x1 = nn.ModuleList()
        self.norms_1 = nn.ModuleList()
        self.norms_2 = nn.ModuleList()

        for i in range(n_layers):
            dilation = kernel_size ** i
            padding = (kernel_size * dilation - dilation) // 2
            self.convs_sep.append(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    groups=channels,
                    dilation=dilation,
                    padding=padding,
                )
            )
            self.convs_1x1.append(nn.Conv1d(channels, channels, 1))
            self.norms_1.append(LayerNorm(channels))
            self.norms_2.append(LayerNorm(channels))

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, g: torch.Tensor | None = None) -> torch.Tensor:
        # g: conditioning [B, C, T] (없으면 None)
        if g is not None:
            x = x + g

        for i in range(self.n_layers):
            y = self.convs_sep[i](x * x_mask)
            y = self.norms_1[i](y)
            y = F.gelu(y)
            y = self.convs_1x1[i](y)
            y = self.norms_2[i](y)
            y = F.gelu(y)
            y = self.drop(y)
            x = x + y

        return x * x_mask


class WN(nn.Module):
    """
    WaveNet-style Dilated Conv 블록.
    - fused_add_tanh_sigmoid_multiply 사용 (commons에 존재)
    - ONNX에서 weight_norm은 export 시 weight_g/weight_v 형태로 남을 수 있어,
      가능하면 export 전에 remove_weight_norm()을 호출하는 것을 권장.
    """
    def __init__(
        self,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
        p_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size는 홀수여야 함"

        self.hidden_channels = hidden_channels
        self.kernel_size = (kernel_size,)
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout

        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        if gin_channels != 0:
            cond_layer = nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1)
            self.cond_layer = weight_norm(cond_layer, name="weight")

        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = int((kernel_size * dilation - dilation) / 2)

            in_layer = nn.Conv1d(
                hidden_channels,
                2 * hidden_channels,
                kernel_size,
                dilation=dilation,
                padding=padding,
            )
            in_layer = weight_norm(in_layer, name="weight")
            self.in_layers.append(in_layer)

            # 마지막 레이어는 skip만 필요
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            res_skip_layer = nn.Conv1d(hidden_channels, res_skip_channels, 1)
            res_skip_layer = weight_norm(res_skip_layer, name="weight")
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, g: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        # output (출력 누적)
        output = torch.zeros_like(x)

        # commons.fused_add_tanh_sigmoid_multiply에서 채널 수 텐서 필요
        # ONNX에서 IntTensor 상수 생성은 대체로 문제 없지만,
        # 아주 빡빡한 런타임이면 buffer로 빼도 됨.
        n_channels_tensor = torch.IntTensor([self.hidden_channels]).to(device=x.device)

        # conditioning (조건) 투영
        if g is not None and self.gin_channels != 0:
            g = self.cond_layer(g)  # [B, 2*H*n_layers, T]

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)

            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset : cond_offset + 2 * self.hidden_channels, :]
            else:
                g_l = torch.zeros_like(x_in)

            acts = commons.fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels_tensor)
            acts = self.drop(acts)

            res_skip_acts = self.res_skip_layers[i](acts)

            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, : self.hidden_channels, :]
                x = (x + res_acts) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels :, :]
            else:
                output = output + res_skip_acts

        return output * x_mask

    def remove_weight_norm(self) -> None:
        if self.gin_channels != 0:
            remove_weight_norm(self.cond_layer)
        for l in self.in_layers:
            remove_weight_norm(l)
        for l in self.res_skip_layers:
            remove_weight_norm(l)


# ============================================================
# HiFiGAN resblocks (decoder parts)
# ============================================================

class ResBlock1(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: tuple[int, int, int] = (1, 3, 5)) -> None:
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0], padding=get_padding(kernel_size, dilation[0]))),
                weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1], padding=get_padding(kernel_size, dilation[1]))),
                weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2], padding=get_padding(kernel_size, dilation[2]))),
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))),
                weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))),
                weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))),
            ]
        )
        self.convs2.apply(init_weights)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor | None = None) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c1(xt)

            xt = F.leaky_relu(xt, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c2(xt)

            x = xt + x

        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self) -> None:
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: tuple[int, int] = (1, 3)) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            [
                weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0], padding=get_padding(kernel_size, dilation[0]))),
                weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1], padding=get_padding(kernel_size, dilation[1]))),
            ]
        )
        self.convs.apply(init_weights)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor | None = None) -> torch.Tensor:
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c(xt)
            x = xt + x

        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self) -> None:
        for l in self.convs:
            remove_weight_norm(l)


# ============================================================
# Flow components (ONNX forward-only)
# - reverse/logdet 제거
# ============================================================

class FlipONNX(nn.Module):
    """
    채널 축 뒤집기.
    - 원본 Flip은 (x, logdet) 반환 + reverse 분기.
    - ONNX용은 그냥 x만 반환.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flip(x, [1])


class LogONNX(nn.Module):
    """
    log 변환 (forward-only).
    - 학습에서 logdet 필요하지만, ONNX 추론에서는 불필요.
    """
    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        y = torch.log(torch.clamp_min(x, 1e-5)) * x_mask
        return y


class ElementwiseAffineONNX(nn.Module):
    """
    y = m + exp(logs) * x (forward-only)
    """
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.m = nn.Parameter(torch.zeros(channels, 1))
        self.logs = nn.Parameter(torch.zeros(channels, 1))

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        y = self.m + torch.exp(self.logs) * x
        return y * x_mask


class ResidualCouplingLayerONNX(nn.Module):
    """
    ResidualCouplingLayer의 forward-only 버전.
    - reverse 제거
    - logdet 제거
    """
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        p_dropout: float = 0.0,
        gin_channels: int = 0,
        mean_only: bool = False,
    ) -> None:
        super().__init__()
        assert channels % 2 == 0, "channels should be divisible by 2"

        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            p_dropout=p_dropout,
            gin_channels=gin_channels,
        )
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - int(mean_only)), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, g: torch.Tensor | None = None) -> torch.Tensor:
        x0, x1 = torch.split(x, [self.half_channels, self.half_channels], dim=1)

        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask

        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels, self.half_channels], dim=1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        x1 = m + x1 * torch.exp(logs) * x_mask
        y = torch.cat([x0, x1], dim=1) * x_mask
        return y


class ConvFlowONNX(nn.Module):
    """
    ConvFlow의 forward-only 버전.
    - reverse 제거
    - logdet 제거
    - transforms는 transforms_onnx.piecewise_rational_quadratic_transform 사용
    """
    def __init__(
        self,
        in_channels: int,
        filter_channels: int,
        kernel_size: int,
        n_layers: int,
        num_bins: int = 10,
        tail_bound: float = 5.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.half_channels = in_channels // 2

        self.pre = nn.Conv1d(self.half_channels, filter_channels, 1)
        self.convs = DDSConv(filter_channels, kernel_size, n_layers, p_dropout=0.0)
        self.proj = nn.Conv1d(filter_channels, self.half_channels * (num_bins * 3 - 1), 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, g: torch.Tensor | None = None) -> torch.Tensor:
        x0, x1 = torch.split(x, [self.half_channels, self.half_channels], dim=1)

        h = self.pre(x0)
        h = self.convs(h, x_mask, g=g)
        h = self.proj(h) * x_mask

        # shape 정리
        # x0: [B, C, T] 여기서 C = half_channels
        b = x0.size(0) # 배치
        c = x0.size(1) # 채널
        t = x0.size(2) # 길이

        # h: [B, C*(num_bins*3-1), T] -> [B, C, T, (num_bins*3-1)]
        h = h.reshape(b, c, -1, t).permute(0, 1, 3, 2)

        unnormalized_widths = h[..., : self.num_bins] / math.sqrt(self.filter_channels)
        unnormalized_heights = h[..., self.num_bins : 2 * self.num_bins] / math.sqrt(self.filter_channels)
        unnormalized_derivatives = h[..., 2 * self.num_bins :]

        # spline transform (forward-only)
        x1, _logabsdet_unused = piecewise_rational_quadratic_transform(
            x1,
            unnormalized_widths,
            unnormalized_heights,
            unnormalized_derivatives,
            inverse=False,
            tails="linear",
            tail_bound=self.tail_bound,
        )

        y = torch.cat([x0, x1], dim=1) * x_mask
        return y


class TransformerCouplingLayerONNX(nn.Module):
    """
    TransformerCouplingLayer의 forward-only 버전.
    - reverse 제거
    - logdet 제거
    """
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        n_layers: int,
        n_heads: int,
        p_dropout: float = 0.0,
        filter_channels: int = 0,
        mean_only: bool = False,
        wn_sharing_parameter=None,
        gin_channels: int = 0,
    ) -> None:
        super().__init__()
        assert channels % 2 == 0, "channels should be divisible by 2"

        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)

        # Encoder는 ONNX에서도 대체로 변환 가능
        self.enc = (
            Encoder(
                hidden_channels,
                filter_channels,
                n_heads,
                n_layers,
                kernel_size,
                p_dropout,
                isflow=True,
                gin_channels=gin_channels,
            )
            if wn_sharing_parameter is None
            else wn_sharing_parameter
        )

        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - int(mean_only)), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, g: torch.Tensor | None = None) -> torch.Tensor:
        x0, x1 = torch.split(x, [self.half_channels, self.half_channels], dim=1)

        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask

        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels, self.half_channels], dim=1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        x1 = m + x1 * torch.exp(logs) * x_mask
        y = torch.cat([x0, x1], dim=1) * x_mask
        return y


# ============================================================
# Flow container (forward-only)
# - 기존 FlowBlock이 (x, logdet) / reverse를 쓰는 구조라면,
#   ONNX에서는 이 컨테이너를 SynthesizerTrnONNX에서 사용하도록 분기한다.
# ============================================================

class ResidualCouplingBlockONNX(nn.Module):
    """
    여러 ResidualCouplingLayerONNX + FlipONNX로 구성되는 forward-only flow 블록.
    """
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        n_flows: int = 4,
        gin_channels: int = 0,
    ) -> None:
        super().__init__()
        self.flows = nn.ModuleList()
        for _ in range(n_flows):
            self.flows.append(
                ResidualCouplingLayerONNX(
                    channels=channels,
                    hidden_channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation_rate=dilation_rate,
                    n_layers=n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                )
            )
            self.flows.append(FlipONNX())

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, g: torch.Tensor | None = None) -> torch.Tensor:
        for flow in self.flows:
            # FlipONNX는 (x)만, Coupling은 (x, x_mask, g)
            if isinstance(flow, FlipONNX):
                x = flow(x)
            else:
                x = flow(x, x_mask, g=g)
        return x


class TransformerCouplingBlockONNX(nn.Module):
    """
    여러 TransformerCouplingLayerONNX + FlipONNX로 구성되는 forward-only flow 블록.
    """
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        n_flows: int = 4,
        gin_channels: int = 0,
        share_parameter: bool = False,
    ) -> None:
        super().__init__()
        self.flows = nn.ModuleList()

        # (옵션) 파라미터 공유: 원본은 wn_sharing_parameter를 쓰지만,
        # ONNX에서는 구현 단순성을 위해 share_parameter=False 권장.
        shared_encoder = None
        if share_parameter:
            shared_encoder = Encoder(
                hidden_channels,
                filter_channels,
                n_heads,
                n_layers,
                kernel_size,
                p_dropout,
                isflow=True,
                gin_channels=gin_channels,
            )

        for _ in range(n_flows):
            self.flows.append(
                TransformerCouplingLayerONNX(
                    channels=channels,
                    hidden_channels=hidden_channels,
                    kernel_size=kernel_size,
                    n_layers=n_layers,
                    n_heads=n_heads,
                    p_dropout=p_dropout,
                    filter_channels=filter_channels,
                    mean_only=True,
                    wn_sharing_parameter=shared_encoder,
                    gin_channels=gin_channels,
                )
            )
            self.flows.append(FlipONNX())

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, g: torch.Tensor | None = None) -> torch.Tensor:
        for flow in self.flows:
            if isinstance(flow, FlipONNX):
                x = flow(x)
            else:
                x = flow(x, x_mask, g=g)
        return x
