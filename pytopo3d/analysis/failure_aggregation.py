"""Smooth global aggregation of nonnegative element failure indices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class FailureAggregateResult:
    """Exact and p-norm summaries over a fixed set of eligible elements."""

    aggregate: float
    raw_pnorm: float
    normalized_pmean: float
    exact_max: float
    correction_factor: float
    exponent: float
    sample_count: int
    weight_sum: float
    element_failure_index: np.ndarray
    eligible_weights: np.ndarray


def _validated_values_and_weights(
    values: np.ndarray,
    *,
    weights: Optional[np.ndarray] = None,
    eligible: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"values must be one-dimensional, got {values.shape}")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("values must be finite and nonnegative")

    if weights is None:
        effective_weights = np.ones(values.shape, dtype=float)
    else:
        effective_weights = np.asarray(weights, dtype=float)
        if effective_weights.shape != values.shape:
            raise ValueError(
                f"weights has shape {effective_weights.shape}, expected {values.shape}"
            )
        if not np.all(np.isfinite(effective_weights)) or np.any(
            effective_weights < 0.0
        ):
            raise ValueError("weights must be finite and nonnegative")

    if eligible is not None:
        eligible = np.asarray(eligible, dtype=bool)
        if eligible.shape != values.shape:
            raise ValueError(
                f"eligible has shape {eligible.shape}, expected {values.shape}"
            )
        effective_weights = np.where(eligible, effective_weights, 0.0)

    if not np.any(effective_weights > 0.0):
        raise ValueError("at least one sample must have positive eligible weight")
    return values, effective_weights


def _validate_exponent(exponent: float) -> float:
    exponent = float(exponent)
    if not np.isfinite(exponent) or exponent <= 1.0:
        raise ValueError(f"exponent must be finite and > 1, got {exponent}")
    return exponent


def pnorm_aggregate(
    values: np.ndarray,
    *,
    exponent: float = 8.0,
    weights: Optional[np.ndarray] = None,
    eligible: Optional[np.ndarray] = None,
    normalized: bool = False,
) -> float:
    """Compute a weighted p-norm or normalized weighted p-mean stably."""
    exponent = _validate_exponent(exponent)
    values, effective_weights = _validated_values_and_weights(
        values,
        weights=weights,
        eligible=eligible,
    )
    positive = effective_weights > 0.0
    scale = float(np.max(values[positive]))
    if scale == 0.0:
        return 0.0

    powered_sum = float(
        np.sum(effective_weights * np.power(values / scale, exponent))
    )
    if normalized:
        powered_sum /= float(np.sum(effective_weights))
    return scale * powered_sum ** (1.0 / exponent)


def calibrate_pnorm_correction(
    values: np.ndarray,
    *,
    exponent: float = 8.0,
    weights: Optional[np.ndarray] = None,
    eligible: Optional[np.ndarray] = None,
    target_maximum: Optional[float] = None,
) -> float:
    """Calibrate a multiplier that matches one fixed reference maximum.

    The returned factor must be frozen while differentiating or optimizing a
    stage. Recomputing it at every perturbed design would collapse the smooth
    aggregate back to the nonsmooth exact maximum.
    """
    values, effective_weights = _validated_values_and_weights(
        values,
        weights=weights,
        eligible=eligible,
    )
    normalized_pmean = pnorm_aggregate(
        values,
        exponent=exponent,
        weights=effective_weights,
        normalized=True,
    )
    if target_maximum is None:
        target_maximum = float(np.max(values[effective_weights > 0.0]))
    else:
        target_maximum = float(target_maximum)
        if not np.isfinite(target_maximum) or target_maximum < 0.0:
            raise ValueError(
                f"target_maximum must be finite and nonnegative, got {target_maximum}"
            )
    if normalized_pmean == 0.0:
        return 1.0
    return target_maximum / normalized_pmean


def corrected_pnorm_aggregate(
    values: np.ndarray,
    *,
    exponent: float = 8.0,
    correction_factor: float = 1.0,
    weights: Optional[np.ndarray] = None,
    eligible: Optional[np.ndarray] = None,
) -> FailureAggregateResult:
    """Return exact, raw, normalized, and corrected p-norm measures."""
    exponent = _validate_exponent(exponent)
    correction_factor = float(correction_factor)
    if not np.isfinite(correction_factor) or correction_factor <= 0.0:
        raise ValueError(
            "correction_factor must be finite and positive, got "
            f"{correction_factor}"
        )
    values, effective_weights = _validated_values_and_weights(
        values,
        weights=weights,
        eligible=eligible,
    )
    raw = pnorm_aggregate(
        values,
        exponent=exponent,
        weights=effective_weights,
        normalized=False,
    )
    normalized = pnorm_aggregate(
        values,
        exponent=exponent,
        weights=effective_weights,
        normalized=True,
    )
    positive = effective_weights > 0.0
    return FailureAggregateResult(
        aggregate=float(correction_factor * normalized),
        raw_pnorm=float(raw),
        normalized_pmean=float(normalized),
        exact_max=float(np.max(values[positive])),
        correction_factor=correction_factor,
        exponent=exponent,
        sample_count=int(np.count_nonzero(positive)),
        weight_sum=float(np.sum(effective_weights)),
        element_failure_index=values.copy(),
        eligible_weights=effective_weights.copy(),
    )


def corrected_pnorm_gradient(result: FailureAggregateResult) -> np.ndarray:
    """Differentiate a corrected normalized p-mean with frozen calibration.

    ``result`` captures the fixed eligible weights and correction factor used
    for the value. The zero-field policy is the zero subgradient. Callers must
    not recalibrate the correction factor during a gradient or finite-
    difference evaluation.
    """
    if not isinstance(result, FailureAggregateResult):
        raise TypeError("result must be a FailureAggregateResult")
    gradient = np.zeros_like(result.element_failure_index, dtype=float)
    if result.normalized_pmean == 0.0:
        return gradient

    positive = (result.eligible_weights > 0.0) & (
        result.element_failure_index > 0.0
    )
    log_gradient = (
        np.log(result.correction_factor)
        + np.log(result.eligible_weights[positive])
        - np.log(result.weight_sum)
        + (result.exponent - 1.0)
        * (
            np.log(result.element_failure_index[positive])
            - np.log(result.normalized_pmean)
        )
    )
    gradient[positive] = np.exp(log_gradient)
    return gradient


def aggregate_gauss_failure(
    failure_index_gauss: np.ndarray,
    *,
    exponent: float = 8.0,
    correction_factor: float = 1.0,
    element_weights: Optional[np.ndarray] = None,
    eligible_elements: Optional[np.ndarray] = None,
) -> FailureAggregateResult:
    """Max-reduce eight H8 Gauss values per element, then aggregate globally."""
    failure_index_gauss = np.asarray(failure_index_gauss, dtype=float)
    if failure_index_gauss.ndim != 2 or failure_index_gauss.shape[1] != 8:
        raise ValueError(
            "failure_index_gauss must have shape (number_of_elements, 8), got "
            f"{failure_index_gauss.shape}"
        )
    if not np.all(np.isfinite(failure_index_gauss)) or np.any(
        failure_index_gauss < 0.0
    ):
        raise ValueError("failure_index_gauss must be finite and nonnegative")
    element_failure_index = np.max(failure_index_gauss, axis=1)
    return corrected_pnorm_aggregate(
        element_failure_index,
        exponent=exponent,
        correction_factor=correction_factor,
        weights=element_weights,
        eligible=eligible_elements,
    )
