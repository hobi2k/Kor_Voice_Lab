from __future__ import annotations

import math
import torch
from torch import nn

from melo import commons
from melo.text.symbols import symbols

from melo.modules_onnx import (
    LayerNorm,
    FlipONNX,
    TransformerCouplingLayerONNX,
    ResidualCouplingLayerONNX,
)

# Decoder: MeloTTS 원본 Generator 사용
from melo.models import Generator
from melo import attentions

# TextEncoderONNX
# - 입력 bert/ja_bert를 [B, T, 768]로 통일 (HF last_hidden_state 그대로)
# - Conv1d 앞에서 transpose 처리
class TextEncoderONNX(nn.Module):
    """
    ONNX 경로용 TextEncoder

    체크포인트 호환성을 위해 원본 TextEncoder의 파라미터 이름/shape를 유지한다.

    - bert_proj:  [hidden_channels, 1024, 1]  (CN계열용)
    - ja_bert_proj:[hidden_channels, 768,  1]  (JP/EN/KR/다국어용)
    """

    def __init__(
        self,
        n_vocab: int,
        out_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        gin_channels: int,
        num_languages: int,
        num_tones: int,
        bert_dim: int = 1024,     # bert_proj 입력 채널
        ja_bert_dim: int = 768,   # ja_bert_proj 입력 채널
    ):
        """
        Args:
            n_vocab:
                입력 토큰 사전 크기. `symbols` 길이와 맞춰야 한다.
            out_channels:
                prior 통계 채널 수. 최종 출력은 `out_channels * 2`로 만들고
                이를 `m`, `logs`로 분리한다.
            hidden_channels:
                텍스트 인코더 내부 기본 채널 폭.
                token/tone/language 임베딩 차원과 encoder 입력 차원으로 쓰인다.
            filter_channels:
                encoder 내부 FFN/conv 계열의 중간 채널 폭.
            n_heads:
                multi-head attention의 head 개수.
            n_layers:
                encoder 레이어 수.
            kernel_size:
                encoder 내부 conv 계열 커널 크기.
            p_dropout:
                encoder 내부 dropout 비율.
            gin_channels:
                speaker embedding 등 글로벌 조건 채널 수.
            num_languages:
                language embedding 테이블 크기.
            num_tones:
                tone embedding 테이블 크기.
            bert_dim:
                `bert_proj` 입력 채널 수.
                checkpoint의 `enc_p.bert_proj.weight` shape와 일치해야 한다.
            ja_bert_dim:
                `ja_bert_proj` 입력 채널 수.
                checkpoint의 `enc_p.ja_bert_proj.weight` shape와 일치해야 한다.
        """
        super().__init__()

        # 기본 설정
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        # Embedding
        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels ** -0.5)

        self.tone_emb = nn.Embedding(num_tones, hidden_channels)
        nn.init.normal_(self.tone_emb.weight, 0.0, hidden_channels ** -0.5)

        self.language_emb = nn.Embedding(num_languages, hidden_channels)
        nn.init.normal_(self.language_emb.weight, 0.0, hidden_channels ** -0.5)

        # BERT projection: shape를 체크포인트와 동일하게 고정
        # enc_p.bert_proj.weight : [hidden_channels, 1024, 1]
        # enc_p.ja_bert_proj.weight : [hidden_channels, 768, 1]
        self.bert_proj = nn.Conv1d(bert_dim, hidden_channels, kernel_size=1)
        self.ja_bert_proj = nn.Conv1d(ja_bert_dim, hidden_channels, kernel_size=1)

        # Encoder(원본 모듈을 그대로 사용해야 state_dict 키가 최대한 동일)
        self.encoder = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            gin_channels=gin_channels,
        )

        # stats projection
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,          # [B, T] (token id)
        x_lengths: torch.Tensor,  # [B]
        tone: torch.Tensor,       # [B, T]
        language: torch.Tensor,   # [B, T]
        bert: torch.Tensor,       # [B, 1024, T] CN계열용 (KR에서는 보통 zeros)
        ja_bert: torch.Tensor,    # [B, 768,  T] KR/JP/EN용 (KR에서는 여기에 kykim 출력)
        g: torch.Tensor | None = None,
    ):
        # 1. token/tone/lang embedding: [B,T,H]
        x_emb = self.emb(x) + self.tone_emb(tone) + self.language_emb(language)

        # 2. bert/ja_bert projection:
        #    bert_proj/ja_bert_proj 입력은 [B,C,T]
        bert_emb = self.bert_proj(bert).transpose(1, 2)        # [B,T,H]
        ja_bert_emb = self.ja_bert_proj(ja_bert).transpose(1, 2)  # [B,T,H]

        # 3. 합산 후 scale
        x_sum = (x_emb + bert_emb + ja_bert_emb) * math.sqrt(self.hidden_channels)

        # 4. [B,H,T] 변환
        x_sum = x_sum.transpose(1, 2)

        # 5. mask
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x_sum.size(2)), 1).to(x_sum.dtype)

        # 6. encoder -> stats
        h = self.encoder(x_sum * x_mask, x_mask, g=g)
        stats = self.proj(h) * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)
        return h, m, logs, x_mask


# DurationPredictorONNX (dp-only)
class DurationPredictorONNX(nn.Module):
    """
    ONNX 추론용 duration predictor.

    텍스트 인코더 출력(`x`)에서 길이(log-duration)를 예측하며,
    출력 shape는 `[B, 1, T]`다.
    """

    def __init__(self, in_channels: int, filter_channels: int, kernel_size: int, p_dropout: float, gin_channels: int = 0):
        """
        Args:
            in_channels:
                입력 feature 채널 수. 보통 `hidden_channels`와 동일하다.
            filter_channels:
                predictor 내부 중간 채널 수.
            kernel_size:
                1D conv 커널 크기.
            p_dropout:
                conv 블록 사이 dropout 비율.
            gin_channels:
                글로벌 조건 채널 수.
                0이 아니면 `cond` conv를 만들고, forward에서 x에 조건을 더한다.
        """
        super().__init__()
        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_1 = LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_2 = LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

        self.gin_channels = gin_channels
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, g: torch.Tensor | None = None):
        if g is not None and self.gin_channels != 0:
            x = x + self.cond(g)

        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)

        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)

        x = self.proj(x * x_mask)
        return x * x_mask  # [B,1,T]


# Flow inverse helpers (weight 공유)
class TransformerCouplingLayerONNXInverse(nn.Module):
    """TransformerCouplingLayerONNX의 역방향(inverse) 계산 래퍼."""

    def __init__(self, forward_layer: TransformerCouplingLayerONNX):
        super().__init__()
        self.fwd = forward_layer

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, g: torch.Tensor | None = None):
        half = self.fwd.half_channels
        x0, x1 = torch.split(x, [half, half], dim=1)

        h = self.fwd.pre(x0) * x_mask
        h = self.fwd.enc(h, x_mask, g=g)
        stats = self.fwd.post(h) * x_mask

        m = stats
        logs = torch.zeros_like(m)
        x1 = (x1 - m) * torch.exp(-logs) * x_mask
        return torch.cat([x0, x1], dim=1) * x_mask


class ResidualCouplingLayerONNXInverse(nn.Module):
    """ResidualCouplingLayerONNX의 역방향(inverse) 계산 래퍼."""

    def __init__(self, forward_layer: ResidualCouplingLayerONNX):
        super().__init__()
        self.fwd = forward_layer

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, g: torch.Tensor | None = None):
        half = self.fwd.half_channels
        x0, x1 = torch.split(x, [half, half], dim=1)

        h = self.fwd.pre(x0) * x_mask
        h = self.fwd.enc(h, x_mask, g=g)
        stats = self.fwd.post(h) * x_mask

        m = stats
        logs = torch.zeros_like(m)
        x1 = (x1 - m) * torch.exp(-logs) * x_mask
        return torch.cat([x0, x1], dim=1) * x_mask


class FlowONNXInverse(nn.Module):
    """forward flow 블록 리스트를 기반으로 reverse 순서 inverse를 수행한다."""

    def __init__(self, flow_layers: nn.ModuleList):
        super().__init__()
        inv_layers = []
        for layer in flow_layers:
            if isinstance(layer, FlipONNX):
                inv_layers.append(layer)
            elif isinstance(layer, TransformerCouplingLayerONNX):
                inv_layers.append(TransformerCouplingLayerONNXInverse(layer))
            elif isinstance(layer, ResidualCouplingLayerONNX):
                inv_layers.append(ResidualCouplingLayerONNXInverse(layer))
            else:
                raise TypeError(f"Unsupported flow layer: {type(layer)}")
        self.inv_layers = nn.ModuleList(inv_layers)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, g: torch.Tensor | None = None):
        for layer in reversed(self.inv_layers):
            if isinstance(layer, FlipONNX):
                x = layer(x)
            else:
                x = layer(x, x_mask, g=g)
        return x


# SynthesizerTrnONNX (dp-only, attn 외부입력 없음)
# - ONNX 추론 단순화를 위해 B=1 경로를 강제한다.
class SynthesizerTrnONNX(nn.Module):
    """MeloTTS 체크포인트를 ONNX 추론 경로로 내보내기 위한 합성기."""

    def __init__(self, hps):
        """
        Args:
            hps:
                MeloTTS 설정 객체(HParams).

                이 클래스에서 사용하는 주요 항목:
                - hps.model.inter_channels:
                    flow/decoder가 다루는 latent 채널 수.
                - hps.model.hidden_channels:
                    text encoder 및 duration predictor의 기본 채널 수.
                - hps.model.filter_channels:
                    encoder/flow 내부 중간 채널 수.
                - hps.model.n_heads, hps.model.n_layers:
                    attention encoder 구조 파라미터.
                - hps.model.kernel_size, hps.model.p_dropout:
                    conv/encoder 공통 하이퍼파라미터.
                - hps.model.n_flow_layer:
                    coupling flow 반복 횟수.
                - hps.model.n_layers_trans_flow:
                    flow 내부 transformer coupling 깊이.
                - hps.model.resblock, resblock_kernel_sizes,
                  resblock_dilation_sizes, upsample_rates,
                  upsample_initial_channel, upsample_kernel_sizes:
                    vocoder(Generator) 구조 파라미터.
                - hps.model.gin_channels:
                    speaker conditioning 채널 수.
                - hps.data.n_speakers:
                    speaker embedding 테이블 크기.
                    0 이하면 sid 경로를 구성할 수 없어 예외를 발생시킨다.
                - hps.num_languages, hps.num_tones:
                    text encoder의 language/tone embedding 크기.

        Notes:
            - ONNX 추론 경로 단순화를 위해 배치 크기 B=1을 전제로 한다.
            - `noise_scale_w`는 입력 시그니처 호환성을 위해 남겨둔 값이며,
              현재 dp-only 경로에서는 직접 사용하지 않는다.
        """
        super().__init__()

        self.inter_channels = hps.model["inter_channels"]
        self.hidden_channels = hps.model["hidden_channels"]
        self.filter_channels = hps.model["filter_channels"]
        self.n_heads = hps.model["n_heads"]
        self.n_layers = hps.model["n_layers"]
        self.kernel_size = hps.model["kernel_size"]
        self.p_dropout = hps.model["p_dropout"]

        self.gin_channels = getattr(hps.model, "gin_channels", 256)
        self.n_speakers = getattr(hps.data, "n_speakers", 0)
        if self.n_speakers <= 0:
            raise RuntimeError("sid 기반(n_speakers>0)만 지원")

        self.emb_g = nn.Embedding(self.n_speakers, self.gin_channels)

        # enc_p의 bert_proj(1024), ja_bert_proj(768) 입력 차원을 고정한다.
        # 이 차원은 checkpoint weight shape와 일치해야 한다.
        self.enc_p = TextEncoderONNX(
            n_vocab=len(symbols),
            out_channels=self.inter_channels,
            hidden_channels=self.hidden_channels,
            filter_channels=self.filter_channels,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            kernel_size=self.kernel_size,
            p_dropout=self.p_dropout,
            gin_channels=self.gin_channels,
            num_languages=hps.num_languages,
            num_tones=hps.num_tones,
            bert_dim=1024,
            ja_bert_dim=768,
        )

        self.dp = DurationPredictorONNX(
            in_channels=self.hidden_channels,
            filter_channels=256,
            kernel_size=3,
            p_dropout=0.5,
            gin_channels=self.gin_channels,
        )

        n_flow_layer = hps.model.get("n_flow_layer", 4)
        n_layers_trans_flow = hps.model.get("n_layers_trans_flow", 6)

        flow_layers = []
        for _ in range(n_flow_layer):
            flow_layers.append(
                TransformerCouplingLayerONNX(
                    channels=self.inter_channels,
                    hidden_channels=self.hidden_channels,
                    kernel_size=5,
                    n_layers=n_layers_trans_flow,
                    n_heads=self.n_heads,
                    p_dropout=self.p_dropout,
                    filter_channels=self.filter_channels,
                    mean_only=True,
                    wn_sharing_parameter=None,
                    gin_channels=self.gin_channels,
                )
            )
            flow_layers.append(FlipONNX())

        # flow 컨테이너 구조를 원본과 동일하게 유지해 state_dict 호환성을 확보한다.
        self.flow = nn.Module()
        self.flow.flows = nn.ModuleList(flow_layers)

        # inverse는 flow.flows를 그대로 참조
        self.flow_inv = FlowONNXInverse(self.flow.flows)

        self.dec = Generator(
            self.inter_channels,
            hps.model["resblock"],
            hps.model["resblock_kernel_sizes"],
            hps.model["resblock_dilation_sizes"],
            hps.model["upsample_rates"],
            hps.model["upsample_initial_channel"],
            hps.model["upsample_kernel_sizes"],
            gin_channels=self.gin_channels,
        )

    def _expand_by_duration(self, feat: torch.Tensor, dur: torch.Tensor):
        """
        feat: [B, C, T_x]
        dur : [B, 1, T_x] (int64)
        return: [1, C, T_y] (B=1 고정)
        """
        B, C, Tx = feat.size()
        if B != 1:
            raise RuntimeError("ONNX export/추론은 B=1만 지원하도록 고정해야 함.")

        rep = dur[0, 0]  # [Tx]
        f = feat[0].transpose(0, 1)               # [Tx, C]
        f_exp = torch.repeat_interleave(f, rep, dim=0)  # [Ty, C]
        return f_exp.transpose(0, 1).unsqueeze(0)       # [1, C, Ty]

    def forward(
        self,
        x: torch.Tensor,          # [B,T]
        x_lengths: torch.Tensor,  # [B]
        sid: torch.Tensor,        # [B]
        tone: torch.Tensor,       # [B,T]
        language: torch.Tensor,   # [B,T]
        bert: torch.Tensor,       # [B,1024,T]
        ja_bert: torch.Tensor,    # [B,768,T]
        noise_scale: torch.Tensor,
        length_scale: torch.Tensor,
        noise_scale_w: torch.Tensor,
    ):
        # noise_scale_w는 모델 입력 시그니처 호환성을 위해 유지한다.
        # 현재 dp-only 경로에서는 직접 사용하지 않는다.
        g = self.emb_g(sid).unsqueeze(-1)  # [B,gin,1]
        x_enc, m_p, logs_p, x_mask = self.enc_p(x, x_lengths, tone, language, bert, ja_bert, g=g)

        logw = self.dp(x_enc, x_mask, g=g)  # [1,1,T_x]
        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w).to(torch.int64)  # [1,1,T_x]

        m_p_exp = self._expand_by_duration(m_p, w_ceil)
        logs_p_exp = self._expand_by_duration(logs_p, w_ceil)

        eps = torch.randn_like(m_p_exp)
        z_p = m_p_exp + eps * torch.exp(logs_p_exp) * noise_scale

        Ty = z_p.size(2)
        y_mask = torch.ones((1, 1, Ty), device=z_p.device, dtype=z_p.dtype)

        z = self.flow_inv(z_p * y_mask, y_mask, g=g)
        audio = self.dec(z * y_mask, g=g)
        return audio
