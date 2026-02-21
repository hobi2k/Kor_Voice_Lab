"""
ONNX 내보내기/실행에 안전한 rational-quadratic spline 변환 구현

핵심 제약:
- numpy 미사용 (torch 연산만 사용)
- 텐서 값에 의존하는 파이썬 분기 최소화
- 함수 호출 시 boolean mask 슬라이싱(x[mask]) 회피
- export를 깨뜨릴 수 있는 in-place 연산 회피
- forward 경로에서 assert/raise에 의존하지 않음
- tail 분기는 torch.where로 처리
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3


def piecewise_rational_quadratic_transform(
    inputs: torch.Tensor,
    unnormalized_widths: torch.Tensor,
    unnormalized_heights: torch.Tensor,
    unnormalized_derivatives: torch.Tensor,
    inverse: bool = False,
    tails: str | None = None,
    tail_bound: float = 1.0,
    min_bin_width: float = DEFAULT_MIN_BIN_WIDTH,
    min_bin_height: float = DEFAULT_MIN_BIN_HEIGHT,
    min_derivative: float = DEFAULT_MIN_DERIVATIVE,
):
    """
    원본 API와 동일한 시그니처를 제공하는 래퍼 함수.

    - tails=None:
      tail 없는 일반 spline 경로를 사용한다.
      (호출 측에서 이미 입력 범위를 [0, 1] 또는 지정 구간으로 맞춘 경우)
    - tails="linear":
      [-tail_bound, tail_bound] 밖에서는 항등함수(선형 tail)로 동작하는
      unconstrained spline 경로를 사용한다.
    """
    if tails is None:
        # VITS 계열에서 자주 쓰는 경로:
        # 호출자가 입력 구간을 미리 정규화했거나 left/right/bottom/top을 명시한 경우.
        outputs, logabsdet = rational_quadratic_spline(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=inverse,
            min_bin_width=min_bin_width,
            min_bin_height=min_bin_height,
            min_derivative=min_derivative,
        )
        return outputs, logabsdet

    # tails가 지정된 경우에는 선형 tail을 포함한 unconstrained 경로를 사용한다.
    outputs, logabsdet = unconstrained_rational_quadratic_spline(
        inputs=inputs,
        unnormalized_widths=unnormalized_widths,
        unnormalized_heights=unnormalized_heights,
        unnormalized_derivatives=unnormalized_derivatives,
        inverse=inverse,
        tails=tails,
        tail_bound=tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
    )
    return outputs, logabsdet


def _searchsorted(bin_locations: torch.Tensor, inputs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    단조 증가하는 bin 경계에서 ONNX 안전하게 인덱스를 찾는 searchsorted 대체 함수.

    주의:
    - export 호환성을 위해 in-place 수정 없이 새 텐서를 만들어 처리한다.
    """
    # 마지막 경계에 eps를 더해 우측 끝 값이 포함되도록 한다.
    last = bin_locations[..., -1:] + eps
    bin_locations_adj = torch.cat([bin_locations[..., :-1], last], dim=-1)

    # 입력보다 작거나 같은 경계 개수를 세고, -1 해서 시작 bin 인덱스를 얻는다.
    # shape: inputs[..., None]와 bin_locations_adj의 브로드캐스팅을 이용한다.
    return torch.sum(inputs.unsqueeze(-1) >= bin_locations_adj, dim=-1) - 1


def unconstrained_rational_quadratic_spline(
    inputs: torch.Tensor,
    unnormalized_widths: torch.Tensor,
    unnormalized_heights: torch.Tensor,
    unnormalized_derivatives: torch.Tensor,
    inverse: bool = False,
    tails: str = "linear",
    tail_bound: float = 1.0,
    min_bin_width: float = DEFAULT_MIN_BIN_WIDTH,
    min_bin_height: float = DEFAULT_MIN_BIN_HEIGHT,
    min_derivative: float = DEFAULT_MIN_DERIVATIVE,
):
    """
    선형 tail을 갖는 unconstrained spline 구현(ONNX 안전 경로)

    구현 포인트:
    - spline 함수 호출 시 boolean mask 슬라이싱을 사용하지 않는다.
    - 입력을 clamp한 뒤 전체 구간에서 spline을 계산하고,
      torch.where로 내부/외부 구간 결과를 선택한다.
      - 내부 구간: spline 결과 사용
      - 외부 구간: 항등함수(선형 tail) 사용
    """
    if tails != "linear":
        # ONNX export 안정성을 위해 linear tail만 지원한다.
        # 실패를 유발하는 예외 대신 안전한 fallback으로 선형 tail을 사용한다.
        tails = "linear"

    # spline이 적용되는 내부 구간 마스크
    inside = (inputs >= -tail_bound) & (inputs <= tail_bound)

    # tail 경계 derivative를 고정 상수로 패딩한다.
    # 목표는 softplus(constant) = 1 - min_derivative를 만족시키는 값이다.
    # constant = log(exp(1 - min_derivative) - 1)
    # 스칼라도 torch 텐서로 만들어 dtype/device를 일치시킨다.
    dtype = unnormalized_derivatives.dtype
    device = unnormalized_derivatives.device

    one = torch.tensor(1.0, dtype=dtype, device=device)
    min_d = torch.tensor(min_derivative, dtype=dtype, device=device)
    constant = torch.log(torch.exp(one - min_d) - one)  # softplus(constant) = 1 - min_derivative

    unnormalized_derivatives_padded = F.pad(unnormalized_derivatives, pad=(1, 1))
    # 일부 백엔드에서 in-place 대입이 export를 불안정하게 만들 수 있어 cat으로 재구성한다.
    left_const = constant.expand_as(unnormalized_derivatives_padded[..., :1])
    right_const = constant.expand_as(unnormalized_derivatives_padded[..., -1:])
    unnormalized_derivatives_padded = torch.cat(
        [
            left_const,
            unnormalized_derivatives_padded[..., 1:-1],
            right_const,
        ],
        dim=-1,
    )

    # 입력을 spline 도메인으로 clamp한 뒤 전체 위치에서 spline을 계산한다.
    inputs_clamped = torch.clamp(inputs, -tail_bound, tail_bound)

    spline_out, spline_logabsdet = rational_quadratic_spline(
        inputs=inputs_clamped,
        unnormalized_widths=unnormalized_widths,
        unnormalized_heights=unnormalized_heights,
        unnormalized_derivatives=unnormalized_derivatives_padded,
        inverse=inverse,
        left=-tail_bound,
        right=tail_bound,
        bottom=-tail_bound,
        top=tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
    )

    # 선형 tail: 외부 구간은 항등함수, log|det J|=0
    zeros = torch.zeros_like(inputs)
    outputs = torch.where(inside, spline_out, inputs)
    logabsdet = torch.where(inside, spline_logabsdet, zeros)
    return outputs, logabsdet


def rational_quadratic_spline(
    inputs: torch.Tensor,
    unnormalized_widths: torch.Tensor,
    unnormalized_heights: torch.Tensor,
    unnormalized_derivatives: torch.Tensor,
    inverse: bool = False,
    left: float = 0.0,
    right: float = 1.0,
    bottom: float = 0.0,
    top: float = 1.0,
    min_bin_width: float = DEFAULT_MIN_BIN_WIDTH,
    min_bin_height: float = DEFAULT_MIN_BIN_HEIGHT,
    min_derivative: float = DEFAULT_MIN_DERIVATIVE,
):
    """
    ONNX 안전 경로의 rational-quadratic spline 본체.

    학습용 구현과의 차이:
    - 런타임 도메인 체크/예외 처리에 의존하지 않는다.
    - forward는 [left, right], inverse는 [bottom, top] 구간 입력을 가정한다.
      (필요하면 호출 측에서 clamp)
    - assert 대신 수치 안전 clamp를 사용한다.
    """
    # export 안정성을 위해 입력 min/max에 대한 강한 검사 로직은 의도적으로 생략한다.

    num_bins = unnormalized_widths.shape[-1]

    # widths: softmax로 양수/합=1을 만든 뒤 최소 bin 폭 제약을 적용한다.
    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1.0 - min_bin_width * float(num_bins)) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
    cumwidths = (right - left) * cumwidths + left

    # 양 끝 경계를 명시적으로 덮어쓴다.
    # in-place를 피하기 위해 텐서를 재구성한다.
    left_t = torch.tensor(left, dtype=cumwidths.dtype, device=cumwidths.device)
    right_t = torch.tensor(right, dtype=cumwidths.dtype, device=cumwidths.device)
    cumwidths = torch.cat(
        [left_t.expand_as(cumwidths[..., :1]), cumwidths[..., 1:-1], right_t.expand_as(cumwidths[..., -1:])],
        dim=-1,
    )
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    # derivatives: softplus로 양수화한 뒤 최소 기울기 제약을 적용한다.
    derivatives = min_derivative + F.softplus(unnormalized_derivatives)

    # heights: softmax 후 최소 bin 높이 제약을 적용한다.
    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1.0 - min_bin_height * float(num_bins)) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
    cumheights = (top - bottom) * cumheights + bottom

    bottom_t = torch.tensor(bottom, dtype=cumheights.dtype, device=cumheights.device)
    top_t = torch.tensor(top, dtype=cumheights.dtype, device=cumheights.device)
    cumheights = torch.cat(
        [bottom_t.expand_as(cumheights[..., :1]), cumheights[..., 1:-1], top_t.expand_as(cumheights[..., -1:])],
        dim=-1,
    )
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    # forward/inverse 모드에 따라 x축/ y축 누적 경계에서 bin 인덱스를 찾는다.
    if inverse:
        # inverse에서는 입력이 y축 값이므로 cumheights 기준으로 탐색한다.
        bin_idx = _searchsorted(cumheights, inputs).unsqueeze(-1)
    else:
        # forward에서는 입력이 x축 값이므로 cumwidths 기준으로 탐색한다.
        bin_idx = _searchsorted(cumwidths, inputs).unsqueeze(-1)

    # 유효 인덱스 범위 [0, num_bins-1]로 clamp한다.
    bin_idx = torch.clamp(bin_idx, 0, num_bins - 1)

    input_cumwidths = cumwidths.gather(-1, bin_idx).squeeze(-1)
    input_bin_widths = widths.gather(-1, bin_idx).squeeze(-1)

    input_cumheights = cumheights.gather(-1, bin_idx).squeeze(-1)

    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx).squeeze(-1)

    input_derivatives = derivatives.gather(-1, bin_idx).squeeze(-1)
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx).squeeze(-1)

    input_heights = heights.gather(-1, bin_idx).squeeze(-1)

    if inverse:
        # theta에 대한 이차방정식을 풀어 inverse 해를 구한다.
        a = (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2.0 * input_delta
        ) + input_heights * (input_delta - input_derivatives)

        b = input_heights * input_derivatives - (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2.0 * input_delta
        )

        c = -input_delta * (inputs - input_cumheights)

        discriminant = b.pow(2) - 4.0 * a * c
        discriminant = torch.clamp(discriminant, min=0.0)

        # 일반 spline 구현에서 쓰는 수치적으로 안정적인 root 선택식을 사용한다.
        root = (2.0 * c) / (-b - torch.sqrt(discriminant) + 1e-12)

        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1.0 - root)
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2.0 * input_delta) * theta_one_minus_theta
        )
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * root.pow(2)
            + 2.0 * input_delta * theta_one_minus_theta
            + input_derivatives * (1.0 - root).pow(2)
        )
        logabsdet = torch.log(derivative_numerator + 1e-12) - 2.0 * torch.log(denominator + 1e-12)
        return outputs, -logabsdet

    # forward 변환
    theta = (inputs - input_cumwidths) / (input_bin_widths + 1e-12)
    theta_one_minus_theta = theta * (1.0 - theta)

    numerator = input_heights * (
        input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta
    )
    denominator = input_delta + (
        (input_derivatives + input_derivatives_plus_one - 2.0 * input_delta) * theta_one_minus_theta
    )
    outputs = input_cumheights + numerator / (denominator + 1e-12)

    derivative_numerator = input_delta.pow(2) * (
        input_derivatives_plus_one * theta.pow(2)
        + 2.0 * input_delta * theta_one_minus_theta
        + input_derivatives * (1.0 - theta).pow(2)
    )
    logabsdet = torch.log(derivative_numerator + 1e-12) - 2.0 * torch.log(denominator + 1e-12)
    return outputs, logabsdet
