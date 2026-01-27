"""Time-series data splitting with Purge/Embargo for leak prevention.

This module implements Train/Val/Test splitting for time-series data
with proper handling of look-ahead bias through Purge and Embargo.

Requirements (PMS-REQ-USDJPY-v5.0 section 14.1):
- Train/Val/Test: Time-series split (no future information)
- Purge/Embargo: max_horizon=60 bars for boundary leak prevention
- Test period: minimum 3 months (calendar days, fixed before evaluation)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Literal, Optional, Sequence, Tuple

from .validation import OhlcvBar

# Constants
MAX_HORIZON_BARS = 60  # Maximum horizon in bars (60 minutes for 1-min bars)
MIN_TEST_DAYS = 90  # Minimum test period in calendar days
DEFAULT_VAL_RATIO = 0.15  # Default validation ratio (from Train+Val)

SplitName = Literal["train", "val", "test"]


class SplitError(ValueError):
    """Exception raised for data splitting errors."""

    pass


@dataclass(frozen=True)
class SplitConfig:
    """Configuration for time-series data splitting.

    Attributes:
        max_horizon_bars: Maximum horizon in bars for Purge/Embargo (default 60)
        min_test_days: Minimum test period in calendar days (default 90)
        val_ratio: Validation ratio from Train+Val (default 0.15)
        test_end_date: Test period end date (default: last bar timestamp)
        test_start_date: Test period start date (default: computed from min_test_days)
    """

    max_horizon_bars: int = MAX_HORIZON_BARS
    min_test_days: int = MIN_TEST_DAYS
    val_ratio: float = DEFAULT_VAL_RATIO
    test_end_date: Optional[datetime] = None
    test_start_date: Optional[datetime] = None

    def __post_init__(self) -> None:
        if self.max_horizon_bars <= 0:
            raise SplitError(
                f"max_horizon_bars must be > 0, got {self.max_horizon_bars}"
            )
        if self.min_test_days <= 0:
            raise SplitError(f"min_test_days must be > 0, got {self.min_test_days}")
        if not (0.0 < self.val_ratio < 1.0):
            raise SplitError(f"val_ratio must be in (0, 1), got {self.val_ratio}")
        if self.test_start_date and self.test_end_date:
            if self.test_start_date >= self.test_end_date:
                raise SplitError("test_start_date must be < test_end_date")


@dataclass(frozen=True)
class SplitBoundary:
    """Boundary information for a single split.

    Attributes:
        name: Split name ("train", "val", or "test")
        start_index: Start index (inclusive)
        end_index: End index (exclusive)
        start_timestamp: Start timestamp
        end_timestamp: End timestamp (of last bar, not exclusive boundary)
        purge_before: Number of bars to exclude at the start (purge)
        purge_after: Number of bars to exclude at the end (purge)
    """

    name: SplitName
    start_index: int
    end_index: int
    start_timestamp: datetime
    end_timestamp: datetime
    purge_before: int
    purge_after: int

    @property
    def total_bars(self) -> int:
        """Total bars in this split (before purge)."""
        return self.end_index - self.start_index

    @property
    def effective_bars(self) -> int:
        """Effective bars after purge exclusion."""
        return max(0, self.total_bars - self.purge_before - self.purge_after)


@dataclass(frozen=True)
class SplitResult:
    """Complete result of time-series splitting.

    Attributes:
        train: Training set boundary
        val: Validation set boundary
        test: Test set boundary
        total_bars: Total number of bars in the dataset
        purge_bars: Total bars excluded by purge (all boundaries)
        embargo_bars: Embargo width in bars (= max_horizon_bars)
        config: Configuration used for splitting
    """

    train: SplitBoundary
    val: SplitBoundary
    test: SplitBoundary
    total_bars: int
    purge_bars: int
    embargo_bars: int
    config: SplitConfig

    def get_train_indices(self) -> Tuple[int, int]:
        """Return (start, end) indices for training data (purge applied).

        Returns:
            Tuple of (start_index, end_index) with purge exclusion applied.
        """
        return (
            self.train.start_index + self.train.purge_before,
            self.train.end_index - self.train.purge_after,
        )

    def get_val_indices(self) -> Tuple[int, int]:
        """Return (start, end) indices for validation data (purge applied).

        Returns:
            Tuple of (start_index, end_index) with purge exclusion applied.
        """
        return (
            self.val.start_index + self.val.purge_before,
            self.val.end_index - self.val.purge_after,
        )

    def get_test_indices(self) -> Tuple[int, int]:
        """Return (start, end) indices for test data (purge applied).

        Returns:
            Tuple of (start_index, end_index) with purge exclusion applied.
        """
        return (
            self.test.start_index + self.test.purge_before,
            self.test.end_index - self.test.purge_after,
        )


def _find_index_by_timestamp(
    bars: Sequence[OhlcvBar], target: datetime, find_first: bool = True
) -> int:
    """Find index of bar by timestamp using binary search.

    Uses manual binary search to avoid O(n) memory allocation from
    extracting all timestamps into a list.

    Args:
        bars: Sorted sequence of OhlcvBar (oldest first)
        target: Target timestamp to find
        find_first: If True, find first bar >= target; if False, find last bar <= target

    Returns:
        Index of the found bar, or len(bars) if not found (for find_first=True)
        or -1 if not found (for find_first=False)
    """
    lo, hi = 0, len(bars)

    if find_first:
        # Find first bar with timestamp >= target (bisect_left equivalent)
        while lo < hi:
            mid = (lo + hi) // 2
            if bars[mid].timestamp_utc < target:
                lo = mid + 1
            else:
                hi = mid
        return lo
    else:
        # Find last bar with timestamp <= target (bisect_right - 1 equivalent)
        while lo < hi:
            mid = (lo + hi) // 2
            if bars[mid].timestamp_utc <= target:
                lo = mid + 1
            else:
                hi = mid
        return lo - 1


def _validate_bars(bars: Sequence[OhlcvBar]) -> None:
    """Validate that bars are properly sorted and non-empty.

    Args:
        bars: Sequence of OhlcvBar to validate

    Raises:
        SplitError: If bars are empty or not strictly increasing
    """
    if not bars:
        raise SplitError("bars sequence is empty")

    prev_ts = None
    for i, bar in enumerate(bars):
        if prev_ts is not None and bar.timestamp_utc <= prev_ts:
            raise SplitError(
                f"bars timestamps must be strictly increasing, "
                f"violation at index {i}: {prev_ts} >= {bar.timestamp_utc}"
            )
        prev_ts = bar.timestamp_utc


def _validate_minimum_data(
    bars: Sequence[OhlcvBar], config: SplitConfig
) -> None:
    """Validate that we have enough data for meaningful splits.

    Args:
        bars: Sequence of OhlcvBar
        config: Split configuration

    Raises:
        SplitError: If insufficient data
    """
    total_days = (bars[-1].timestamp_utc - bars[0].timestamp_utc).days

    # Need at least 2x test period (Train + Test minimum)
    min_total_days = config.min_test_days * 2
    if total_days < min_total_days:
        raise SplitError(
            f"Insufficient data: need at least {min_total_days} calendar days, "
            f"got {total_days} days"
        )

    # Check minimum bars for purge regions (4 boundaries x max_horizon)
    min_bars = config.max_horizon_bars * 6  # Extra margin
    if len(bars) < min_bars:
        raise SplitError(
            f"Insufficient bars: need at least {min_bars}, got {len(bars)}"
        )


def split_timeseries(
    bars: Sequence[OhlcvBar],
    config: Optional[SplitConfig] = None,
) -> SplitResult:
    """Split time-series data into Train/Val/Test with Purge/Embargo.

    This function performs time-series splitting while ensuring:
    - Chronological order is preserved (no shuffling)
    - Future information is blocked (no look-ahead bias)
    - Purge regions exclude bars near boundaries to prevent label leakage
    - Test period is at least min_test_days (calendar days)

    Args:
        bars: OHLCV bars sequence (must be sorted by timestamp, oldest first)
        config: Split configuration. If None, uses default settings

    Returns:
        SplitResult containing boundaries and indices for each split

    Raises:
        SplitError: If insufficient data or invalid configuration

    Example:
        >>> from pms.splitting import split_timeseries, SplitConfig
        >>> config = SplitConfig(min_test_days=90, val_ratio=0.15)
        >>> result = split_timeseries(bars, config)
        >>> train_start, train_end = result.get_train_indices()
        >>> train_bars = bars[train_start:train_end]
    """
    if config is None:
        config = SplitConfig()

    # Validate input
    _validate_bars(bars)
    _validate_minimum_data(bars, config)

    total_bars = len(bars)
    first_timestamp = bars[0].timestamp_utc
    last_timestamp = bars[-1].timestamp_utc

    # Determine test period boundaries (calendar days based)
    if config.test_end_date:
        test_end_date = config.test_end_date
    else:
        test_end_date = last_timestamp

    if config.test_start_date:
        test_start_date = config.test_start_date
    else:
        # Calculate test start date from min_test_days (calendar days)
        test_start_date = test_end_date - timedelta(days=config.min_test_days)

    # Validate test period
    test_duration_days = (test_end_date - test_start_date).days
    if test_duration_days < config.min_test_days:
        raise SplitError(
            f"Test period must be at least {config.min_test_days} days, "
            f"got {test_duration_days} days"
        )

    if test_start_date <= first_timestamp:
        raise SplitError(
            f"Test start date {test_start_date} is before or at data start "
            f"{first_timestamp}. Need more historical data."
        )

    # Find test set boundaries (by index)
    test_start_index = _find_index_by_timestamp(bars, test_start_date, find_first=True)
    test_end_index = total_bars  # End is exclusive, so use total_bars

    if test_start_index >= test_end_index:
        raise SplitError(
            f"No bars found in test period [{test_start_date}, {test_end_date}]"
        )

    # Train+Val is everything before test
    train_val_end_index = test_start_index

    # Split Train and Val based on val_ratio
    train_val_bars = train_val_end_index
    val_bars_count = int(train_val_bars * config.val_ratio)
    train_end_index = train_val_end_index - val_bars_count
    val_start_index = train_end_index
    val_end_index = train_val_end_index

    # Validate we have enough bars for each split after purge
    purge_bars = config.max_horizon_bars
    min_effective_bars = purge_bars * 2  # Need some effective bars after purge

    if train_end_index - purge_bars < min_effective_bars:
        raise SplitError(
            f"Train set too small after purge: "
            f"{train_end_index} bars, need at least {min_effective_bars + purge_bars}"
        )

    if (val_end_index - val_start_index) - (purge_bars * 2) < min_effective_bars:
        raise SplitError(
            f"Val set too small after purge: "
            f"{val_end_index - val_start_index} bars, need at least "
            f"{min_effective_bars + purge_bars * 2}"
        )

    if (test_end_index - test_start_index) - purge_bars < min_effective_bars:
        raise SplitError(
            f"Test set too small after purge: "
            f"{test_end_index - test_start_index} bars, need at least "
            f"{min_effective_bars + purge_bars}"
        )

    # Create boundary objects with purge information
    train_boundary = SplitBoundary(
        name="train",
        start_index=0,
        end_index=train_end_index,
        start_timestamp=bars[0].timestamp_utc,
        end_timestamp=bars[train_end_index - 1].timestamp_utc,
        purge_before=0,  # No purge at data start
        purge_after=purge_bars,  # Purge before Val boundary
    )

    val_boundary = SplitBoundary(
        name="val",
        start_index=val_start_index,
        end_index=val_end_index,
        start_timestamp=bars[val_start_index].timestamp_utc,
        end_timestamp=bars[val_end_index - 1].timestamp_utc,
        purge_before=purge_bars,  # Purge after Train boundary
        purge_after=purge_bars,  # Purge before Test boundary
    )

    test_boundary = SplitBoundary(
        name="test",
        start_index=test_start_index,
        end_index=test_end_index,
        start_timestamp=bars[test_start_index].timestamp_utc,
        end_timestamp=bars[test_end_index - 1].timestamp_utc,
        purge_before=purge_bars,  # Purge after Val boundary
        purge_after=0,  # No purge at data end
    )

    # Calculate total purge bars for reporting purposes.
    # Note: This counts purge regions from each split's perspective, meaning
    # boundary regions are counted twice (Train.purge_after and Val.purge_before
    # refer to the same bars, as do Val.purge_after and Test.purge_before).
    # This is intentional for reporting "total exclusion impact" across splits.
    # Actual unique bars excluded = purge_bars * 2 (one region per boundary).
    total_purge_bars = purge_bars * 4

    return SplitResult(
        train=train_boundary,
        val=val_boundary,
        test=test_boundary,
        total_bars=total_bars,
        purge_bars=total_purge_bars,
        embargo_bars=purge_bars,
        config=config,
    )


def get_split_bars(
    bars: Sequence[OhlcvBar],
    split_result: SplitResult,
    split_name: SplitName,
    apply_purge: bool = True,
) -> List[OhlcvBar]:
    """Extract bars for a specific split.

    Args:
        bars: Original OHLCV bars sequence (same as passed to split_timeseries)
        split_result: Result from split_timeseries
        split_name: Which split to extract ("train", "val", "test")
        apply_purge: Whether to exclude purge regions (default True)

    Returns:
        List of OhlcvBar for the specified split

    Raises:
        SplitError: If invalid split_name

    Example:
        >>> train_bars = get_split_bars(bars, result, "train")
        >>> val_bars = get_split_bars(bars, result, "val")
        >>> test_bars = get_split_bars(bars, result, "test")
    """
    if split_name == "train":
        if apply_purge:
            start, end = split_result.get_train_indices()
        else:
            start = split_result.train.start_index
            end = split_result.train.end_index
    elif split_name == "val":
        if apply_purge:
            start, end = split_result.get_val_indices()
        else:
            start = split_result.val.start_index
            end = split_result.val.end_index
    elif split_name == "test":
        if apply_purge:
            start, end = split_result.get_test_indices()
        else:
            start = split_result.test.start_index
            end = split_result.test.end_index
    else:
        raise SplitError(f"Invalid split_name: {split_name}")

    return list(bars[start:end])


def validate_split_for_similarity_search(
    query_index: int,
    candidate_indices: Sequence[int],
    split_result: SplitResult,
    lookback_bars: int,
) -> List[int]:
    """Filter candidate indices for similarity search with embargo.

    This function implements the embargo requirement from section 11.4:
    "Query windows overlapping with candidates are excluded (embargo=60 bars)"

    The embargo ensures that:
    1. Candidate's judgment window doesn't overlap with query's lookback window
    2. Query's judgment window doesn't overlap with candidate's lookback window
    3. Same timestamp is excluded (scenario_end == t)

    Args:
        query_index: Index of query bar (scenario_end_t)
        candidate_indices: Candidate bar indices from Train+Val
        split_result: Split result for embargo configuration
        lookback_bars: Lookback window size (query window length)

    Returns:
        Filtered list of valid candidate indices

    Example:
        >>> candidates = list(range(result.train.start_index, result.val.end_index))
        >>> valid = validate_split_for_similarity_search(
        ...     query_index=50000,
        ...     candidate_indices=candidates,
        ...     split_result=result,
        ...     lookback_bars=512
        ... )
    """
    max_horizon = split_result.embargo_bars
    valid_indices: List[int] = []

    for candidate_idx in candidate_indices:
        if _is_embargoed(query_index, candidate_idx, lookback_bars, max_horizon):
            continue
        valid_indices.append(candidate_idx)

    return valid_indices


def _is_embargoed(
    query_t: int, candidate_t: int, lookback_bars: int, max_horizon: int
) -> bool:
    """Check if candidate overlaps with query window considering embargo.

    Args:
        query_t: Query bar index (scenario end)
        candidate_t: Candidate bar index (scenario end)
        lookback_bars: Lookback window size
        max_horizon: Maximum horizon for judgment window

    Returns:
        True if candidate should be excluded (embargoed), False otherwise
    """
    # Same timestamp is always excluded
    if candidate_t == query_t:
        return True

    # Query lookback window: [query_t - lookback_bars + 1, query_t]
    query_lookback_start = query_t - lookback_bars + 1
    query_lookback_end = query_t

    # Query judgment window: [query_t + 1, query_t + max_horizon]
    query_judgment_start = query_t + 1
    query_judgment_end = query_t + max_horizon

    # Candidate lookback window: [candidate_t - lookback_bars + 1, candidate_t]
    candidate_lookback_start = candidate_t - lookback_bars + 1
    candidate_lookback_end = candidate_t

    # Candidate judgment window: [candidate_t + 1, candidate_t + max_horizon]
    candidate_judgment_start = candidate_t + 1
    candidate_judgment_end = candidate_t + max_horizon

    # Check if candidate's judgment window overlaps with query's lookback window
    if (
        candidate_judgment_start <= query_lookback_end
        and candidate_judgment_end >= query_lookback_start
    ):
        return True

    # Check if query's judgment window overlaps with candidate's lookback window
    if (
        query_judgment_start <= candidate_lookback_end
        and query_judgment_end >= candidate_lookback_start
    ):
        return True

    return False


def get_split_summary(split_result: SplitResult) -> str:
    """Generate a human-readable summary of the split.

    Args:
        split_result: Result from split_timeseries

    Returns:
        Multi-line string with split statistics

    Example output:
        Split Summary:
        - Train: 2024-01-01 ~ 2025-10-01 (920,000 bars -> 919,940 effective)
        - Val:   2025-10-01 ~ 2026-01-01 (40,000 bars -> 39,880 effective)
        - Test:  2026-01-01 ~ 2026-04-01 (130,000 bars -> 129,940 effective)
        - Total: 1,090,000 bars
        - Purge: 240 bars (60 bars x 4 boundaries)
        - Embargo: 60 bars (for similarity search)
    """

    def fmt_ts(ts: datetime) -> str:
        return ts.strftime("%Y-%m-%d %H:%M")

    def fmt_num(n: int) -> str:
        return f"{n:,}"

    lines = [
        "Split Summary:",
        f"- Train: {fmt_ts(split_result.train.start_timestamp)} ~ "
        f"{fmt_ts(split_result.train.end_timestamp)} "
        f"({fmt_num(split_result.train.total_bars)} bars -> "
        f"{fmt_num(split_result.train.effective_bars)} effective)",
        f"- Val:   {fmt_ts(split_result.val.start_timestamp)} ~ "
        f"{fmt_ts(split_result.val.end_timestamp)} "
        f"({fmt_num(split_result.val.total_bars)} bars -> "
        f"{fmt_num(split_result.val.effective_bars)} effective)",
        f"- Test:  {fmt_ts(split_result.test.start_timestamp)} ~ "
        f"{fmt_ts(split_result.test.end_timestamp)} "
        f"({fmt_num(split_result.test.total_bars)} bars -> "
        f"{fmt_num(split_result.test.effective_bars)} effective)",
        f"- Total: {fmt_num(split_result.total_bars)} bars",
        f"- Purge: {fmt_num(split_result.purge_bars)} bars "
        f"({split_result.embargo_bars} bars x 4 boundaries)",
        f"- Embargo: {fmt_num(split_result.embargo_bars)} bars (for similarity search)",
    ]

    return "\n".join(lines)
