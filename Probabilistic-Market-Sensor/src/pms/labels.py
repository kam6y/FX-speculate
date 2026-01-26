from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Sequence

from .validation import OhlcvBar

# Constants
HORIZONS = [10, 30, 60]  # minutes (bars)
BARRIER_MULTIPLIER = 0.5
ATR_WINDOW = 20

LabelClass = Literal["up", "down", "choppy", "neutral"]


class LabelError(ValueError):
    """Exception raised for label generation errors."""
    pass


@dataclass(frozen=True)
class BarrierConfig:
    """Configuration for label generation."""
    horizons: List[int] = field(default_factory=lambda: HORIZONS.copy())
    atr_window: int = ATR_WINDOW
    barrier_multiplier: float = BARRIER_MULTIPLIER

    def __post_init__(self) -> None:
        if self.atr_window <= 0:
            raise LabelError(f"atr_window must be > 0, got {self.atr_window}")
        if self.barrier_multiplier <= 0:
            raise LabelError(f"barrier_multiplier must be > 0, got {self.barrier_multiplier}")
        if not self.horizons:
            raise LabelError("horizons must not be empty")
        for h in self.horizons:
            if h <= 0:
                raise LabelError(f"horizon must be > 0, got {h}")


@dataclass(frozen=True)
class Label:
    """Label for a single horizon."""
    horizon_min: int
    label_class: LabelClass
    barrier_upper: float
    barrier_lower: float
    first_hit_bar: int  # relative to t (1-indexed in judgment window)
    hit_type: str  # "upper", "lower", "both", "none"


@dataclass(frozen=True)
class LabelResult:
    """Complete label result for all horizons."""
    labels: List[Label]
    base_close: float
    atr20: float


def generate_labels(
    bars: Sequence[OhlcvBar],
    horizons: List[int] = None,
    config: BarrierConfig = None,
) -> LabelResult:
    """
    Generate 4-class labels for multiple horizons using ATR20-based barriers.

    Args:
        bars: OHLCV bars sequence (must include at least ATR_WINDOW bars for
              base calculation + max(horizons) bars for judgment window)
        horizons: List of horizon lengths in minutes (bars). Defaults to [10, 30, 60]
        config: Barrier configuration. If None, uses default settings

    Returns:
        LabelResult containing labels for each horizon

    Raises:
        LabelError: If insufficient data, invalid values, or calculation fails

    Notes:
        - bars[t] is the reference bar (last bar before judgment window)
        - bars[t-ATR_WINDOW+1:t+1] used for ATR20 calculation
        - bars[t+1:t+H+1] is the judgment window for horizon H
        - Barriers: C_t ± (barrier_multiplier × ATR20)
        - Label classes:
            * "up": Only upper barrier hit
            * "down": Only lower barrier hit
            * "choppy": Both barriers hit (any order)
            * "neutral": Neither barrier hit
    """
    if config is None:
        config = BarrierConfig()
    if horizons is None:
        horizons = config.horizons

    # Validation
    max_horizon = max(horizons)
    min_required = config.atr_window + max_horizon

    if len(bars) < min_required:
        raise LabelError(
            f"at least {min_required} bars required "
            f"(atr_window={config.atr_window} + max_horizon={max_horizon}), "
            f"got {len(bars)}"
        )

    # Calculate ATR20 at reference point (t)
    t = len(bars) - max_horizon - 1
    atr20 = _calculate_atr20(bars, t, config.atr_window)
    base_close = bars[t].close

    # Calculate barriers
    barrier_upper = base_close + config.barrier_multiplier * atr20
    barrier_lower = base_close - config.barrier_multiplier * atr20

    # Generate labels for each horizon
    labels = []
    for horizon in sorted(horizons):
        label = _generate_single_label(
            bars, t, horizon, barrier_upper, barrier_lower
        )
        labels.append(label)

    return LabelResult(
        labels=labels,
        base_close=base_close,
        atr20=atr20,
    )


def _calculate_true_range(
    current_bar: OhlcvBar,
    previous_close: float,
) -> float:
    """
    Calculate True Range for a single bar.

    Formula: max(high-low, |high-prev_close|, |low-prev_close|)

    Args:
        current_bar: Current OHLCV bar
        previous_close: Previous bar's close price

    Returns:
        True Range value
    """
    hl_range = current_bar.high - current_bar.low
    high_prev_close = abs(current_bar.high - previous_close)
    low_prev_close = abs(current_bar.low - previous_close)

    return max(hl_range, high_prev_close, low_prev_close)


def _calculate_atr20(
    bars: Sequence[OhlcvBar],
    t: int,
    window: int,
) -> float:
    """
    Calculate ATR20 using Simple Moving Average of True Range.

    Args:
        bars: OHLCV bars sequence
        t: Index of reference bar
        window: ATR window size (typically 20)

    Returns:
        ATR value (SMA of True Range over window)

    Raises:
        LabelError: If insufficient data or invalid values
    """
    if t < window - 1:
        raise LabelError(
            f"insufficient bars for ATR{window}: "
            f"need at least {window} bars ending at index {t}"
        )

    start_idx = t - window + 1

    # Validate all bars in window
    for i in range(start_idx, t + 1):
        bar = bars[i]
        if not (bar.low <= bar.close <= bar.high and
                bar.low <= bar.open <= bar.high):
            raise LabelError(
                f"invalid OHLC at index {i}: high={bar.high}, "
                f"low={bar.low}, open={bar.open}, close={bar.close}"
            )
        if bar.high < bar.low:
            raise LabelError(
                f"invalid bar at index {i}: high {bar.high} < low {bar.low}"
            )

    # Calculate True Range for each bar in window
    tr_sum = 0.0
    for i in range(start_idx, t + 1):
        if i == 0:
            # First bar: True Range = high - low
            tr_sum += bars[i].high - bars[i].low
        else:
            prev_close = bars[i - 1].close
            tr_sum += _calculate_true_range(bars[i], prev_close)

    atr = tr_sum / window

    if atr <= 0:
        raise LabelError(f"ATR must be > 0, got {atr}")

    return atr


def _check_barrier_hit(
    bar: OhlcvBar,
    barrier_upper: float,
    barrier_lower: float,
) -> tuple[bool, bool]:
    """
    Check if a bar hits upper/lower barriers.

    Args:
        bar: OHLCV bar to check
        barrier_upper: Upper barrier price
        barrier_lower: Lower barrier price

    Returns:
        (upper_hit, lower_hit) tuple of booleans
    """
    upper_hit = bar.high >= barrier_upper
    lower_hit = bar.low <= barrier_lower
    return upper_hit, lower_hit


def _generate_single_label(
    bars: Sequence[OhlcvBar],
    t: int,
    horizon: int,
    barrier_upper: float,
    barrier_lower: float,
) -> Label:
    """
    Generate label for a single horizon.

    Args:
        bars: OHLCV bars sequence
        t: Reference bar index
        horizon: Horizon length in bars
        barrier_upper: Upper barrier price
        barrier_lower: Lower barrier price

    Returns:
        Label with classification and metadata
    """
    # Judgment window: bars[t+1] to bars[t+horizon] (inclusive)
    window_start = t + 1
    window_end = t + horizon

    if window_end >= len(bars):
        raise LabelError(
            f"insufficient bars for horizon {horizon}: "
            f"need index {window_end}, have {len(bars)} bars"
        )

    upper_hit = False
    lower_hit = False
    first_hit_bar = None
    hit_type = "none"

    for i in range(window_start, window_end + 1):
        bar = bars[i]
        u_hit, l_hit = _check_barrier_hit(bar, barrier_upper, barrier_lower)

        if u_hit and not upper_hit:
            upper_hit = True
            if first_hit_bar is None:
                first_hit_bar = i - t
                hit_type = "upper"

        if l_hit and not lower_hit:
            lower_hit = True
            if first_hit_bar is None:
                first_hit_bar = i - t
                hit_type = "lower"

        # Both hit
        if upper_hit and lower_hit:
            hit_type = "both"
            break

    # Determine label class
    if upper_hit and lower_hit:
        label_class = "choppy"
    elif upper_hit:
        label_class = "up"
    elif lower_hit:
        label_class = "down"
    else:
        label_class = "neutral"
        first_hit_bar = 0  # No hit
        hit_type = "none"

    return Label(
        horizon_min=horizon,
        label_class=label_class,
        barrier_upper=barrier_upper,
        barrier_lower=barrier_lower,
        first_hit_bar=first_hit_bar or 0,
        hit_type=hit_type,
    )
