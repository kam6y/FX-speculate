"""Baseline prediction models for direction and range forecasting.

This module implements simple baseline predictors that serve as comparison
benchmarks for the main model. These baselines establish the minimum
performance thresholds that the main model must exceed.

Baselines implemented:
- Baseline-A (Direction): Always predicts 'neutral'
- Baseline-B (Direction): Predicts based on 1-minute return sign
- Range Baseline: Uses fixed quantiles from training distribution
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Sequence

from .validation import OhlcvBar
from .labels import HORIZONS

# Constants
RETURN_THRESHOLD_PCT = 0.0002  # 0.02% threshold for Baseline-B neutral zone
QUANTILES = [0.10, 0.50, 0.90]

DirectionPrediction = Literal["up", "down", "neutral", "choppy"]


class BaselineError(ValueError):
    """Exception raised for baseline generation errors."""

    pass


@dataclass(frozen=True)
class BaselineConfig:
    """Configuration for baseline predictors.

    Attributes:
        horizons: List of horizons in minutes (default [10, 30, 60])
        return_threshold_pct: Threshold for Baseline-B neutral zone (default 0.0002)
        quantiles: Quantiles for range baseline (default [0.10, 0.50, 0.90])
    """

    horizons: List[int] = field(default_factory=lambda: HORIZONS.copy())
    return_threshold_pct: float = RETURN_THRESHOLD_PCT
    quantiles: List[float] = field(default_factory=lambda: QUANTILES.copy())

    def __post_init__(self) -> None:
        if not self.horizons:
            raise BaselineError("horizons must not be empty")
        for h in self.horizons:
            if h <= 0:
                raise BaselineError(f"horizon must be > 0, got {h}")
        if self.return_threshold_pct < 0:
            raise BaselineError(
                f"return_threshold_pct must be >= 0, got {self.return_threshold_pct}"
            )
        if len(self.quantiles) != 3:
            raise BaselineError("quantiles must have exactly 3 elements")
        if self.quantiles != sorted(self.quantiles):
            raise BaselineError("quantiles must be sorted ascending")


@dataclass(frozen=True)
class DirectionPredictionResult:
    """Result of a direction baseline prediction.

    Attributes:
        horizon_min: Horizon in minutes
        predicted_class: Predicted label class
        probabilities: Probability distribution over 4 classes
    """

    horizon_min: int
    predicted_class: DirectionPrediction
    probabilities: Dict[str, float]


@dataclass(frozen=True)
class RangePredictionResult:
    """Result of a range baseline prediction.

    Attributes:
        horizon_min: Horizon in minutes
        q10: 10th percentile price prediction
        q50: 50th percentile (median) price prediction
        q90: 90th percentile price prediction
    """

    horizon_min: int
    q10: float
    q50: float
    q90: float


@dataclass(frozen=True)
class TrainedRangeBaseline:
    """Trained range baseline with fixed quantile multipliers.

    These multipliers are computed from training data and represent
    the H-minute return distribution quantiles.

    Attributes:
        horizon_min: Horizon in minutes
        return_q10: 10th percentile of H-minute returns from training
        return_q50: 50th percentile of H-minute returns from training
        return_q90: 90th percentile of H-minute returns from training
    """

    horizon_min: int
    return_q10: float
    return_q50: float
    return_q90: float


# =============================================================================
# Baseline-A: Always predict neutral
# =============================================================================


def predict_baseline_a(
    horizon: int,
) -> DirectionPredictionResult:
    """
    Baseline-A: Always predict 'neutral'.

    This is the simplest possible baseline that always predicts the
    same class regardless of input.

    Args:
        horizon: Horizon in minutes

    Returns:
        DirectionPredictionResult with neutral prediction
    """
    return DirectionPredictionResult(
        horizon_min=horizon,
        predicted_class="neutral",
        probabilities={"up": 0.0, "down": 0.0, "neutral": 1.0, "choppy": 0.0},
    )


def predict_baseline_a_batch(
    n_samples: int,
    horizons: List[int],
) -> List[List[DirectionPredictionResult]]:
    """
    Baseline-A predictions for multiple samples.

    Args:
        n_samples: Number of samples
        horizons: List of horizons in minutes

    Returns:
        List of prediction results per sample, each containing results per horizon
    """
    return [[predict_baseline_a(h) for h in sorted(horizons)] for _ in range(n_samples)]


# =============================================================================
# Baseline-B: 1-minute return sign with neutral zone
# =============================================================================


def compute_return_1(
    prev_close: float,
    close: float,
) -> float:
    """
    Compute 1-minute return (close_t / close_{t-1} - 1).

    Args:
        prev_close: Previous bar's close price
        close: Current bar's close price

    Returns:
        Return as a decimal (not percentage)

    Raises:
        BaselineError: If prev_close is zero or negative
    """
    if prev_close <= 0:
        raise BaselineError(f"prev_close must be > 0, got {prev_close}")
    return (close / prev_close) - 1.0


def predict_baseline_b(
    return_1: float,
    horizon: int,
    threshold: float = RETURN_THRESHOLD_PCT,
) -> DirectionPredictionResult:
    """
    Baseline-B: Predict based on 1-minute return sign.

    - If |return_1| < threshold: predict neutral
    - If return_1 >= threshold: predict up
    - If return_1 <= -threshold: predict down

    Note: This baseline never predicts 'choppy'.

    Args:
        return_1: 1-minute return (decimal, not percentage)
        horizon: Horizon in minutes
        threshold: Neutral zone threshold (default 0.0002 = 0.02%)

    Returns:
        DirectionPredictionResult with predicted class
    """
    abs_return = abs(return_1)

    if abs_return < threshold:
        predicted: DirectionPrediction = "neutral"
    elif return_1 >= 0:
        predicted = "up"
    else:
        predicted = "down"

    # Create deterministic probability distribution
    probs = {"up": 0.0, "down": 0.0, "neutral": 0.0, "choppy": 0.0}
    probs[predicted] = 1.0

    return DirectionPredictionResult(
        horizon_min=horizon,
        predicted_class=predicted,
        probabilities=probs,
    )


def predict_baseline_b_from_bars(
    bars: Sequence[OhlcvBar],
    horizons: List[int],
    config: BaselineConfig | None = None,
) -> List[DirectionPredictionResult]:
    """
    Baseline-B prediction from OHLCV bars.

    Uses the last two bars to compute the 1-minute return and
    predicts based on its sign.

    Args:
        bars: Sequence of OHLCV bars (at least 2 bars required)
        horizons: List of horizons in minutes
        config: Baseline configuration (optional)

    Returns:
        List of predictions for each horizon (sorted ascending)

    Raises:
        BaselineError: If insufficient bars
    """
    if config is None:
        config = BaselineConfig(horizons=horizons)

    if len(bars) < 2:
        raise BaselineError("at least 2 bars required for Baseline-B")

    return_1 = compute_return_1(bars[-2].close, bars[-1].close)

    return [
        predict_baseline_b(return_1, h, config.return_threshold_pct)
        for h in sorted(horizons)
    ]


# =============================================================================
# Range Baseline: Fixed quantile from training distribution
# =============================================================================


def train_range_baseline(
    bars: Sequence[OhlcvBar],
    horizon: int,
) -> TrainedRangeBaseline:
    """
    Train range baseline by computing H-minute return quantiles.

    The baseline uses training period returns to establish fixed quantile
    multipliers. At inference, it predicts: close_t * (1 + q_p) for each
    quantile p.

    Args:
        bars: Training OHLCV bars (sorted chronologically, oldest first)
        horizon: Horizon in minutes (bars)

    Returns:
        TrainedRangeBaseline with quantile multipliers

    Raises:
        BaselineError: If insufficient bars or no valid returns
    """
    if len(bars) < horizon + 1:
        raise BaselineError(f"at least {horizon + 1} bars required, got {len(bars)}")

    # Compute H-minute returns: (close_{t+H} / close_t) - 1
    returns = []
    for i in range(len(bars) - horizon):
        close_t = bars[i].close
        close_t_plus_h = bars[i + horizon].close
        if close_t > 0:
            ret = (close_t_plus_h / close_t) - 1.0
            returns.append(ret)

    if not returns:
        raise BaselineError("no valid returns computed")

    # Sort returns for quantile computation
    sorted_returns = sorted(returns)
    n = len(sorted_returns)

    def quantile(q: float) -> float:
        """Compute quantile using linear interpolation."""
        pos = q * (n - 1)
        lower_idx = int(math.floor(pos))
        upper_idx = int(math.ceil(pos))
        if lower_idx == upper_idx:
            return sorted_returns[lower_idx]
        frac = pos - lower_idx
        return sorted_returns[lower_idx] * (1 - frac) + sorted_returns[upper_idx] * frac

    return TrainedRangeBaseline(
        horizon_min=horizon,
        return_q10=quantile(0.10),
        return_q50=quantile(0.50),
        return_q90=quantile(0.90),
    )


def predict_range_baseline(
    current_close: float,
    trained_baseline: TrainedRangeBaseline,
) -> RangePredictionResult:
    """
    Predict range using trained baseline.

    Prediction formula: close_t * (1 + q_p) for each quantile p.

    Args:
        current_close: Current close price (close_t)
        trained_baseline: Trained range baseline parameters

    Returns:
        RangePredictionResult with q10/q50/q90 price predictions
    """
    return RangePredictionResult(
        horizon_min=trained_baseline.horizon_min,
        q10=current_close * (1 + trained_baseline.return_q10),
        q50=current_close * (1 + trained_baseline.return_q50),
        q90=current_close * (1 + trained_baseline.return_q90),
    )


def train_range_baselines(
    bars: Sequence[OhlcvBar],
    horizons: List[int] = None,
) -> Dict[int, TrainedRangeBaseline]:
    """
    Train range baselines for multiple horizons.

    Args:
        bars: Training OHLCV bars (sorted chronologically, oldest first)
        horizons: List of horizons in minutes (default [10, 30, 60])

    Returns:
        Dict mapping horizon to TrainedRangeBaseline

    Raises:
        BaselineError: If insufficient bars for any horizon
    """
    if horizons is None:
        horizons = HORIZONS.copy()

    return {h: train_range_baseline(bars, h) for h in horizons}
