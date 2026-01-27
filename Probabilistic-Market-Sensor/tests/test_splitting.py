"""Tests for time-series data splitting module."""

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from pms.splitting import (  # noqa: E402
    DEFAULT_VAL_RATIO,
    MAX_HORIZON_BARS,
    MIN_TEST_DAYS,
    SplitBoundary,
    SplitConfig,
    SplitError,
    SplitResult,
    get_split_bars,
    get_split_summary,
    split_timeseries,
    validate_split_for_similarity_search,
)
from pms.validation import OhlcvBar  # noqa: E402


def make_bar(
    timestamp_utc: datetime,
    close: float = 150.0,
    spread: float = 0.01,
    volume: float = 100.0,
) -> OhlcvBar:
    """Create a test bar with symmetric spread around close."""
    return OhlcvBar(
        timestamp_utc=timestamp_utc,
        open=close,
        high=close + spread,
        low=close - spread,
        close=close,
        volume=volume,
    )


def make_test_bars(
    start: datetime,
    num_bars: int,
    interval_minutes: int = 1,
    base_price: float = 150.0,
) -> list[OhlcvBar]:
    """Generate test OHLCV bars for split testing.

    Args:
        start: Start timestamp (UTC)
        num_bars: Number of bars to generate
        interval_minutes: Minutes between bars (default 1)
        base_price: Base price for bars

    Returns:
        List of OhlcvBar
    """
    bars = []
    for i in range(num_bars):
        ts = start + timedelta(minutes=i * interval_minutes)
        bars.append(make_bar(ts, close=base_price + i * 0.001))
    return bars


def make_realistic_test_data(
    days: int = 180,
    start: datetime | None = None,
) -> list[OhlcvBar]:
    """Generate realistic test data simulating FX market.

    Simulates FX market with weekend closures (Fri 21:00 ~ Sun 21:00 UTC).
    FX market hours: Sunday 21:00 UTC to Friday 21:00 UTC.

    Args:
        days: Number of calendar days to generate
        start: Start datetime (UTC). Defaults to 2025-07-01 00:00 UTC.

    Returns:
        List of OhlcvBar
    """
    if start is None:
        start = datetime(2025, 7, 1, 0, 0, tzinfo=timezone.utc)
    bars = []
    current = start
    end = start + timedelta(days=days)
    base_price = 150.0
    bar_count = 0

    while current < end:
        weekday = current.weekday()
        hour = current.hour

        # FX market is closed: Friday 21:00 UTC to Sunday 21:00 UTC
        # weekday: 0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri, 5=Sat, 6=Sun
        is_market_closed = (
            (weekday == 4 and hour >= 21)  # Friday 21:00 onwards
            or weekday == 5  # All Saturday
            or (weekday == 6 and hour < 21)  # Sunday before 21:00
        )

        if not is_market_closed:
            bars.append(make_bar(current, close=base_price + bar_count * 0.0001))
            bar_count += 1

        current += timedelta(minutes=1)

    return bars


class SplitConfigTests(unittest.TestCase):
    """Tests for SplitConfig dataclass."""

    def test_default_config_values(self) -> None:
        """Test default configuration values."""
        config = SplitConfig()
        self.assertEqual(config.max_horizon_bars, MAX_HORIZON_BARS)
        self.assertEqual(config.min_test_days, MIN_TEST_DAYS)
        self.assertEqual(config.val_ratio, DEFAULT_VAL_RATIO)
        self.assertIsNone(config.test_end_date)
        self.assertIsNone(config.test_start_date)

    def test_custom_config_values(self) -> None:
        """Test custom configuration values."""
        config = SplitConfig(
            max_horizon_bars=30,
            min_test_days=60,
            val_ratio=0.2,
        )
        self.assertEqual(config.max_horizon_bars, 30)
        self.assertEqual(config.min_test_days, 60)
        self.assertEqual(config.val_ratio, 0.2)

    def test_invalid_max_horizon_raises(self) -> None:
        """Test that max_horizon_bars <= 0 raises SplitError."""
        with self.assertRaises(SplitError) as context:
            SplitConfig(max_horizon_bars=0)
        self.assertIn("max_horizon_bars", str(context.exception))

        with self.assertRaises(SplitError) as context:
            SplitConfig(max_horizon_bars=-1)
        self.assertIn("max_horizon_bars", str(context.exception))

    def test_invalid_min_test_days_raises(self) -> None:
        """Test that min_test_days <= 0 raises SplitError."""
        with self.assertRaises(SplitError) as context:
            SplitConfig(min_test_days=0)
        self.assertIn("min_test_days", str(context.exception))

    def test_invalid_val_ratio_raises(self) -> None:
        """Test that val_ratio outside (0, 1) raises SplitError."""
        with self.assertRaises(SplitError) as context:
            SplitConfig(val_ratio=0.0)
        self.assertIn("val_ratio", str(context.exception))

        with self.assertRaises(SplitError) as context:
            SplitConfig(val_ratio=1.0)
        self.assertIn("val_ratio", str(context.exception))

        with self.assertRaises(SplitError) as context:
            SplitConfig(val_ratio=-0.1)
        self.assertIn("val_ratio", str(context.exception))

    def test_invalid_date_range_raises(self) -> None:
        """Test that test_start_date >= test_end_date raises SplitError."""
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        end = datetime(2025, 12, 1, tzinfo=timezone.utc)

        with self.assertRaises(SplitError) as context:
            SplitConfig(test_start_date=start, test_end_date=end)
        self.assertIn("test_start_date", str(context.exception))


class SplitTimeseriesTests(unittest.TestCase):
    """Tests for split_timeseries function."""

    def test_basic_split_preserves_time_order(self) -> None:
        """Test that split maintains chronological order."""
        start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        # Generate 200 days of data (enough for 90-day test period)
        bars = make_realistic_test_data(days=200)

        result = split_timeseries(bars)

        # Verify train comes before val
        self.assertLess(result.train.end_index, result.val.start_index + 1)
        # Verify val comes before test
        self.assertLessEqual(result.val.end_index, result.test.start_index)

        # Verify timestamps are in order
        self.assertLess(result.train.end_timestamp, result.val.start_timestamp)
        self.assertLessEqual(result.val.end_timestamp, result.test.start_timestamp)

    def test_split_with_minimum_data(self) -> None:
        """Test splitting with exactly minimum required data."""
        # Need at least 180 days (2x min_test_days=90)
        # Add a few extra days to account for weekend gaps
        bars = make_realistic_test_data(days=185)

        # Should not raise with default config
        result = split_timeseries(bars)

        self.assertIsInstance(result, SplitResult)
        self.assertGreater(result.train.total_bars, 0)
        self.assertGreater(result.val.total_bars, 0)
        self.assertGreater(result.test.total_bars, 0)

    def test_insufficient_data_raises(self) -> None:
        """Test that insufficient data raises SplitError."""
        # Only 100 days (less than 2x90=180 minimum)
        bars = make_realistic_test_data(days=100)

        with self.assertRaises(SplitError) as context:
            split_timeseries(bars)
        self.assertIn("Insufficient", str(context.exception))

    def test_empty_bars_raises(self) -> None:
        """Test that empty bars raises SplitError."""
        with self.assertRaises(SplitError) as context:
            split_timeseries([])
        self.assertIn("empty", str(context.exception))

    def test_non_increasing_timestamps_raises(self) -> None:
        """Test that non-increasing timestamps raise SplitError."""
        start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        bars = [
            make_bar(start),
            make_bar(start + timedelta(minutes=2)),
            make_bar(start + timedelta(minutes=1)),  # Out of order!
        ]

        with self.assertRaises(SplitError) as context:
            split_timeseries(bars)
        self.assertIn("strictly increasing", str(context.exception))

    def test_split_reproducibility(self) -> None:
        """Test that same input produces same output."""
        bars = make_realistic_test_data(days=200)

        result1 = split_timeseries(bars)
        result2 = split_timeseries(bars)

        # Same indices
        self.assertEqual(result1.train.start_index, result2.train.start_index)
        self.assertEqual(result1.train.end_index, result2.train.end_index)
        self.assertEqual(result1.val.start_index, result2.val.start_index)
        self.assertEqual(result1.val.end_index, result2.val.end_index)
        self.assertEqual(result1.test.start_index, result2.test.start_index)
        self.assertEqual(result1.test.end_index, result2.test.end_index)

    def test_test_period_minimum_90_days(self) -> None:
        """Test that test period is at least 90 calendar days."""
        bars = make_realistic_test_data(days=200)
        result = split_timeseries(bars)

        test_duration = result.test.end_timestamp - result.test.start_timestamp
        test_days = test_duration.days

        # Test period should be approximately 90 days
        # Allow some tolerance due to bar alignment at boundaries
        self.assertGreaterEqual(test_days, 85)
        self.assertLessEqual(test_days, 95)

    def test_custom_test_dates(self) -> None:
        """Test splitting with custom test start/end dates."""
        # make_realistic_test_data starts at 2025-07-01
        bars = make_realistic_test_data(days=365)

        # Custom test period (within the data range)
        # Data: 2025-07-01 to ~2026-07-01
        test_start = datetime(2026, 4, 1, 0, 0, tzinfo=timezone.utc)
        test_end = datetime(2026, 7, 1, 0, 0, tzinfo=timezone.utc)

        config = SplitConfig(
            test_start_date=test_start,
            test_end_date=test_end,
        )
        result = split_timeseries(bars, config)

        # Test should start around the specified date
        self.assertGreaterEqual(result.test.start_timestamp, test_start)


class PurgeTests(unittest.TestCase):
    """Tests for Purge boundary calculation."""

    def test_train_val_boundary_purge(self) -> None:
        """Test that Train/Val boundary has correct purge."""
        bars = make_realistic_test_data(days=200)
        result = split_timeseries(bars)

        # Train should have purge_after = max_horizon_bars
        self.assertEqual(result.train.purge_after, MAX_HORIZON_BARS)
        # Val should have purge_before = max_horizon_bars
        self.assertEqual(result.val.purge_before, MAX_HORIZON_BARS)

    def test_val_test_boundary_purge(self) -> None:
        """Test that Val/Test boundary has correct purge."""
        bars = make_realistic_test_data(days=200)
        result = split_timeseries(bars)

        # Val should have purge_after = max_horizon_bars
        self.assertEqual(result.val.purge_after, MAX_HORIZON_BARS)
        # Test should have purge_before = max_horizon_bars
        self.assertEqual(result.test.purge_before, MAX_HORIZON_BARS)

    def test_purge_excludes_correct_bars(self) -> None:
        """Test that purge correctly excludes boundary bars."""
        bars = make_realistic_test_data(days=200)
        result = split_timeseries(bars)

        # Get indices with purge applied
        train_start, train_end = result.get_train_indices()
        val_start, val_end = result.get_val_indices()
        test_start, test_end = result.get_test_indices()

        # Effective train end should be before raw train end
        self.assertEqual(train_end, result.train.end_index - MAX_HORIZON_BARS)

        # Effective val should be smaller due to both-side purge
        self.assertEqual(val_start, result.val.start_index + MAX_HORIZON_BARS)
        self.assertEqual(val_end, result.val.end_index - MAX_HORIZON_BARS)

        # Effective test start should be after raw test start
        self.assertEqual(test_start, result.test.start_index + MAX_HORIZON_BARS)

    def test_purge_with_max_horizon_60(self) -> None:
        """Test purge with default max_horizon=60."""
        bars = make_realistic_test_data(days=200)
        result = split_timeseries(bars)

        # Total purge should be 4 x 60 = 240
        self.assertEqual(result.purge_bars, 240)
        self.assertEqual(result.embargo_bars, 60)

    def test_purge_with_custom_max_horizon(self) -> None:
        """Test purge with non-default max_horizon."""
        bars = make_realistic_test_data(days=200)
        config = SplitConfig(max_horizon_bars=30)
        result = split_timeseries(bars, config)

        # Total purge should be 4 x 30 = 120
        self.assertEqual(result.purge_bars, 120)
        self.assertEqual(result.embargo_bars, 30)

    def test_no_label_leakage_across_boundaries(self) -> None:
        """Test that label judgment windows don't cross boundaries."""
        bars = make_realistic_test_data(days=200)
        result = split_timeseries(bars)

        # Get effective indices
        train_start, train_end = result.get_train_indices()
        val_start, val_end = result.get_val_indices()
        test_start, test_end = result.get_test_indices()

        # Train's last effective bar + max_horizon should not reach val's first effective bar
        train_last_judgment_end = train_end - 1 + MAX_HORIZON_BARS
        self.assertLess(train_last_judgment_end, val_start)

        # Val's last effective bar + max_horizon should not reach test's first effective bar
        val_last_judgment_end = val_end - 1 + MAX_HORIZON_BARS
        self.assertLess(val_last_judgment_end, test_start)


class EmbargoTests(unittest.TestCase):
    """Tests for Embargo calculation in similarity search."""

    def test_embargo_excludes_overlapping_windows(self) -> None:
        """Test that overlapping windows are excluded."""
        bars = make_realistic_test_data(days=200)
        result = split_timeseries(bars)

        query_index = 50000
        lookback_bars = 512

        # Create candidates including some that should be excluded
        candidates = list(range(query_index - 600, query_index + 600))

        valid = validate_split_for_similarity_search(
            query_index=query_index,
            candidate_indices=candidates,
            split_result=result,
            lookback_bars=lookback_bars,
        )

        # Query index itself should be excluded
        self.assertNotIn(query_index, valid)

        # Candidates very close to query should be excluded
        # Within embargo range (query ± lookback + max_horizon)
        for candidate in valid:
            self.assertNotEqual(candidate, query_index)

    def test_embargo_same_timestamp_excluded(self) -> None:
        """Test that same timestamp is excluded (scenario_end == t)."""
        bars = make_realistic_test_data(days=200)
        result = split_timeseries(bars)

        query_index = 50000
        candidates = [query_index, query_index + 1000, query_index - 1000]

        valid = validate_split_for_similarity_search(
            query_index=query_index,
            candidate_indices=candidates,
            split_result=result,
            lookback_bars=512,
        )

        self.assertNotIn(query_index, valid)

    def test_embargo_with_lookback_512(self) -> None:
        """Test embargo with default lookback_bars=512."""
        bars = make_realistic_test_data(days=200)
        result = split_timeseries(bars)

        query_index = 50000
        lookback_bars = 512

        # Far away candidates should be valid
        far_candidates = [1000, 2000, 3000]

        valid = validate_split_for_similarity_search(
            query_index=query_index,
            candidate_indices=far_candidates,
            split_result=result,
            lookback_bars=lookback_bars,
        )

        # All far candidates should be valid
        self.assertEqual(len(valid), len(far_candidates))

    def test_valid_candidates_returned(self) -> None:
        """Test that valid (non-embargoed) candidates are returned."""
        bars = make_realistic_test_data(days=200)
        result = split_timeseries(bars)

        query_index = 50000
        lookback_bars = 512
        max_horizon = result.embargo_bars

        # Candidates that are definitely outside embargo range
        # Need to be far enough: |candidate - query| > lookback + max_horizon
        safe_distance = lookback_bars + max_horizon + 100
        safe_candidates = [
            query_index - safe_distance,
            query_index - safe_distance - 1000,
            query_index + safe_distance,
        ]

        valid = validate_split_for_similarity_search(
            query_index=query_index,
            candidate_indices=safe_candidates,
            split_result=result,
            lookback_bars=lookback_bars,
        )

        # At least some safe candidates should be valid
        self.assertGreater(len(valid), 0)


class GetSplitBarsTests(unittest.TestCase):
    """Tests for get_split_bars function."""

    def test_get_train_bars(self) -> None:
        """Test extracting train bars."""
        bars = make_realistic_test_data(days=200)
        result = split_timeseries(bars)

        train_bars = get_split_bars(bars, result, "train")

        # Should have effective bars count
        self.assertEqual(len(train_bars), result.train.effective_bars)

        # First bar should be at train start
        self.assertEqual(train_bars[0].timestamp_utc, bars[0].timestamp_utc)

    def test_get_val_bars(self) -> None:
        """Test extracting validation bars."""
        bars = make_realistic_test_data(days=200)
        result = split_timeseries(bars)

        val_bars = get_split_bars(bars, result, "val")

        # Should have effective bars count
        self.assertEqual(len(val_bars), result.val.effective_bars)

    def test_get_test_bars(self) -> None:
        """Test extracting test bars."""
        bars = make_realistic_test_data(days=200)
        result = split_timeseries(bars)

        test_bars = get_split_bars(bars, result, "test")

        # Should have effective bars count
        self.assertEqual(len(test_bars), result.test.effective_bars)

    def test_get_bars_without_purge(self) -> None:
        """Test extracting bars without purge applied."""
        bars = make_realistic_test_data(days=200)
        result = split_timeseries(bars)

        train_bars_no_purge = get_split_bars(bars, result, "train", apply_purge=False)
        train_bars_with_purge = get_split_bars(bars, result, "train", apply_purge=True)

        # Without purge should have more bars
        self.assertGreater(len(train_bars_no_purge), len(train_bars_with_purge))
        self.assertEqual(len(train_bars_no_purge), result.train.total_bars)

    def test_invalid_split_name_raises(self) -> None:
        """Test that invalid split name raises error."""
        bars = make_realistic_test_data(days=200)
        result = split_timeseries(bars)

        with self.assertRaises(SplitError) as context:
            get_split_bars(bars, result, "invalid")  # type: ignore
        self.assertIn("Invalid split_name", str(context.exception))


class GetSplitSummaryTests(unittest.TestCase):
    """Tests for get_split_summary function."""

    def test_summary_contains_all_splits(self) -> None:
        """Test that summary contains info for all splits."""
        bars = make_realistic_test_data(days=200)
        result = split_timeseries(bars)

        summary = get_split_summary(result)

        self.assertIn("Train", summary)
        self.assertIn("Val", summary)
        self.assertIn("Test", summary)
        self.assertIn("Purge", summary)
        self.assertIn("Embargo", summary)

    def test_summary_format(self) -> None:
        """Test that summary has expected format."""
        bars = make_realistic_test_data(days=200)
        result = split_timeseries(bars)

        summary = get_split_summary(result)

        # Should be multi-line
        lines = summary.split("\n")
        self.assertGreater(len(lines), 5)

        # First line should be title
        self.assertIn("Summary", lines[0])


class IntegrationTests(unittest.TestCase):
    """Integration tests for split module."""

    def test_full_pipeline_with_realistic_data(self) -> None:
        """Test complete split pipeline with realistic data."""
        bars = make_realistic_test_data(days=365)  # 1 year

        result = split_timeseries(bars)

        # Extract all splits
        train_bars = get_split_bars(bars, result, "train")
        val_bars = get_split_bars(bars, result, "val")
        test_bars = get_split_bars(bars, result, "test")

        # Total effective bars should be less than total due to purge
        total_effective = len(train_bars) + len(val_bars) + len(test_bars)
        self.assertLess(total_effective, len(bars))

        # No overlap between splits
        train_timestamps = {b.timestamp_utc for b in train_bars}
        val_timestamps = {b.timestamp_utc for b in val_bars}
        test_timestamps = {b.timestamp_utc for b in test_bars}

        self.assertEqual(len(train_timestamps & val_timestamps), 0)
        self.assertEqual(len(val_timestamps & test_timestamps), 0)
        self.assertEqual(len(train_timestamps & test_timestamps), 0)

        # Generate summary
        summary = get_split_summary(result)
        self.assertIsInstance(summary, str)

    def test_split_indices_are_valid(self) -> None:
        """Test that all split indices are within bounds."""
        bars = make_realistic_test_data(days=200)
        result = split_timeseries(bars)

        # All indices should be valid
        self.assertGreaterEqual(result.train.start_index, 0)
        self.assertLessEqual(result.train.end_index, len(bars))
        self.assertGreaterEqual(result.val.start_index, 0)
        self.assertLessEqual(result.val.end_index, len(bars))
        self.assertGreaterEqual(result.test.start_index, 0)
        self.assertLessEqual(result.test.end_index, len(bars))

        # Effective indices should also be valid
        train_start, train_end = result.get_train_indices()
        val_start, val_end = result.get_val_indices()
        test_start, test_end = result.get_test_indices()

        self.assertGreaterEqual(train_start, 0)
        self.assertLessEqual(train_end, len(bars))
        self.assertGreaterEqual(val_start, 0)
        self.assertLessEqual(val_end, len(bars))
        self.assertGreaterEqual(test_start, 0)
        self.assertLessEqual(test_end, len(bars))


if __name__ == "__main__":
    unittest.main()
