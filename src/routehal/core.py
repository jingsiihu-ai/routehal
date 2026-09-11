"""Numerical building blocks for expert-routing divergence.

The implementation is model-agnostic and uses NumPy so it can be tested without
downloading model weights. A model adapter only needs to convert captured router
and attention tensors to arrays before calling :func:`compute_hroute`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.floating]


@dataclass(frozen=True)
class HRouteResult:
    """RouteHal scores for a batch of generation steps."""

    per_example: FloatArray
    per_layer: FloatArray


def _as_distribution(values: ArrayLike, *, axis: int = -1, eps: float = 1e-8) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("probability inputs must contain only finite values")
    if np.any(array < 0):
        raise ValueError("probability inputs must be non-negative")

    totals = array.sum(axis=axis, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("every probability vector must have positive mass")

    normalized = array / totals
    normalized = np.clip(normalized, eps, None)
    return normalized / normalized.sum(axis=axis, keepdims=True)


def js_divergence(p: ArrayLike, q: ArrayLike, *, eps: float = 1e-8) -> FloatArray:
    """Return Jensen-Shannon divergence normalized to the interval [0, 1]."""

    p_dist = _as_distribution(p, eps=eps)
    q_dist = _as_distribution(q, eps=eps)
    if p_dist.shape != q_dist.shape:
        raise ValueError(f"distribution shapes must match, got {p_dist.shape} and {q_dist.shape}")

    midpoint = 0.5 * (p_dist + q_dist)
    kl_p = np.sum(p_dist * (np.log(p_dist) - np.log(midpoint)), axis=-1)
    kl_q = np.sum(q_dist * (np.log(q_dist) - np.log(midpoint)), axis=-1)
    return np.clip(0.5 * (kl_p + kl_q) / np.log(2.0), 0.0, 1.0)


def visual_routing_prototype(
    visual_router_probs: ArrayLike,
    visual_attention: ArrayLike,
    *,
    eps: float = 1e-8,
) -> FloatArray:
    """Build the attention-weighted routing prototype for each layer and batch item.

    Args:
        visual_router_probs: Array shaped ``[layers, batch, patches, experts]``.
        visual_attention: Array shaped ``[layers, batch, patches]``.
    """

    routing = _as_distribution(visual_router_probs, eps=eps)
    attention = _as_distribution(visual_attention, eps=eps)

    if routing.ndim != 4:
        raise ValueError("visual_router_probs must have shape [layers, batch, patches, experts]")
    if attention.ndim != 3:
        raise ValueError("visual_attention must have shape [layers, batch, patches]")
    if routing.shape[:-1] != attention.shape:
        raise ValueError(
            "visual routing and attention must agree on [layers, batch, patches], "
            f"got {routing.shape[:-1]} and {attention.shape}"
        )

    prototype = np.sum(attention[..., None] * routing, axis=-2)
    return _as_distribution(prototype, eps=eps)


def compute_hroute(
    text_router_probs: ArrayLike,
    visual_router_probs: ArrayLike,
    visual_attention: ArrayLike,
    *,
    layer_weights: ArrayLike | None = None,
    calibration_mean: float | ArrayLike | None = None,
    calibration_std: float | ArrayLike | None = None,
    eps: float = 1e-8,
) -> HRouteResult:
    """Compute per-layer and cross-layer routing divergence scores.

    Args:
        text_router_probs: ``[layers, batch, experts]`` routing probabilities for
            the current prediction state.
        visual_router_probs: ``[layers, batch, patches, experts]`` routing
            probabilities for visual tokens.
        visual_attention: ``[layers, batch, patches]`` attention mass from the
            current prediction state to visual tokens.
        layer_weights: Optional non-negative weights shaped ``[layers]``.
        calibration_mean: Optional held-out mean used for z-score calibration.
        calibration_std: Optional held-out standard deviation. Must be supplied
            together with ``calibration_mean`` and be strictly positive.
    """

    text = _as_distribution(text_router_probs, eps=eps)
    if text.ndim != 3:
        raise ValueError("text_router_probs must have shape [layers, batch, experts]")

    prototype = visual_routing_prototype(visual_router_probs, visual_attention, eps=eps)
    if text.shape != prototype.shape:
        raise ValueError(f"text routing and visual prototype shapes must match, got {text.shape} and {prototype.shape}")

    per_layer = js_divergence(text, prototype, eps=eps)
    num_layers = per_layer.shape[0]

    if layer_weights is None:
        weights = np.full(num_layers, 1.0 / num_layers, dtype=np.float64)
    else:
        weights = np.asarray(layer_weights, dtype=np.float64)
        if weights.shape != (num_layers,):
            raise ValueError(f"layer_weights must have shape ({num_layers},)")
        if np.any(~np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("layer_weights must be finite, non-negative, and have positive mass")
        weights = weights / weights.sum()

    per_example = np.sum(weights[:, None] * per_layer, axis=0)

    if (calibration_mean is None) != (calibration_std is None):
        raise ValueError("calibration_mean and calibration_std must be supplied together")
    if calibration_mean is not None and calibration_std is not None:
        std = np.asarray(calibration_std, dtype=np.float64)
        if np.any(~np.isfinite(std)) or np.any(std <= 0):
            raise ValueError("calibration_std must be finite and strictly positive")
        per_example = (per_example - np.asarray(calibration_mean, dtype=np.float64)) / std

    return HRouteResult(per_example=per_example, per_layer=per_layer)
