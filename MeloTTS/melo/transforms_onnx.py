"""
ONNX-safe rational-quadratic spline transforms (BV2-style)

Key constraints:
- NO numpy
- NO data-dependent Python branching on tensors
- NO boolean mask slicing (x[mask]) for function calls
- NO in-place ops that break export (e.g., bin_locations[..., -1] += eps)
- NO asserts/raises in the forward path
- Uses torch.where for tails masking
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
    Wrapper that matches the original API.
    - tails is either None (no tails; domain is [0,1] by default in spline caller)
      or "linear" (unconstrained spline with linear tails on [-tail_bound, tail_bound]).
    """
    if tails is None:
        # In many VITS-style flows, this path is used when the caller already normalizes
        # into [0,1] (or provides left/right/bottom/top explicitly).
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

    # tails != None: use unconstrained with tails
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
    ONNX-safe searchsorted for monotonic bin_locations.

    Important: avoid in-place modification of bin_locations.
    """
    # Add eps to the last bin edge to ensure the rightmost edge is included.
    last = bin_locations[..., -1:] + eps
    bin_locations_adj = torch.cat([bin_locations[..., :-1], last], dim=-1)

    # Count how many bin edges <= input, then -1 to get index of bin start.
    # Shape: inputs[..., None] broadcast against bin_locations_adj
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
    ONNX-safe unconstrained spline with linear tails on [-tail_bound, tail_bound].

    Implementation notes:
    - Avoid boolean mask slicing into spline() call.
    - Compute spline everywhere on a clamped version, then use torch.where to select:
        inside: spline result
        outside: identity (linear tails)
    """
    if tails != "linear":
        # Keep only linear tails for ONNX-safe export.
        # (No raise: do a safe fallback to linear to avoid export-time failure)
        tails = "linear"

    # Mask of inside interval
    inside = (inputs >= -tail_bound) & (inputs <= tail_bound)

    # Pad derivatives for tails: set boundary derivatives to a constant that maps to (1 - min_derivative)
    # Original: constant = log(exp(1 - min_derivative) - 1)
    # Use torch-only scalar
    dtype = unnormalized_derivatives.dtype
    device = unnormalized_derivatives.device

    one = torch.tensor(1.0, dtype=dtype, device=device)
    min_d = torch.tensor(min_derivative, dtype=dtype, device=device)
    constant = torch.log(torch.exp(one - min_d) - one)  # softplus(constant) = 1 - min_derivative

    unnormalized_derivatives_padded = F.pad(unnormalized_derivatives, pad=(1, 1))
    # Avoid in-place assign that may confuse export in some cases:
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

    # Clamp inputs into spline domain; compute spline everywhere
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

    # Linear tails: outside interval -> identity, logabsdet=0
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
    ONNX-safe rational-quadratic spline.

    Differences from training version:
    - No runtime domain checks / raises.
    - Inputs are assumed to be within [left, right] for forward
      and within [bottom, top] for inverse, so callers should clamp if needed.
    - No assert; discriminant is clamped >= 0 for numerical safety.
    """
    # NOTE: we intentionally do not check min/max against [left,right] to keep export-safe.

    num_bins = unnormalized_widths.shape[-1]

    # widths: softmax -> enforce min_bin_width
    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1.0 - min_bin_width * float(num_bins)) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
    cumwidths = (right - left) * cumwidths + left

    # Explicitly set endpoints (avoid in-place ops by rebuilding)
    left_t = torch.tensor(left, dtype=cumwidths.dtype, device=cumwidths.device)
    right_t = torch.tensor(right, dtype=cumwidths.dtype, device=cumwidths.device)
    cumwidths = torch.cat(
        [left_t.expand_as(cumwidths[..., :1]), cumwidths[..., 1:-1], right_t.expand_as(cumwidths[..., -1:])],
        dim=-1,
    )
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    # derivatives: softplus -> enforce min_derivative
    derivatives = min_derivative + F.softplus(unnormalized_derivatives)

    # heights: softmax -> enforce min_bin_height
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

    # Choose bin by searching in cumwidths or cumheights
    if inverse:
        # inputs are in y-space, so locate in cumheights
        bin_idx = _searchsorted(cumheights, inputs).unsqueeze(-1)
    else:
        # inputs are in x-space, so locate in cumwidths
        bin_idx = _searchsorted(cumwidths, inputs).unsqueeze(-1)

    # Clamp bin indices to valid range [0, num_bins-1] (ONNX-safe)
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
        # Solve quadratic for theta (root in [0,1])
        a = (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2.0 * input_delta
        ) + input_heights * (input_delta - input_derivatives)

        b = input_heights * input_derivatives - (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2.0 * input_delta
        )

        c = -input_delta * (inputs - input_cumheights)

        discriminant = b.pow(2) - 4.0 * a * c
        discriminant = torch.clamp(discriminant, min=0.0)

        # Numerically stable root selection used in common spline implementations
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

    # Forward transform
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
