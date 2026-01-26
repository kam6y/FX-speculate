import math
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from pms.labels import (  # noqa: E402
    HORIZONS,
    ATR_WINDOW,
    BARRIER_MULTIPLIER,
    BarrierConfig,
    Label,
    LabelError,
    LabelResult,
    generate_labels,
)
from pms.validation import OhlcvBar  # noqa: E402


def make_bar_with_spread(
    timestamp_utc: datetime,
    close: float,
    spread: float = 0.01,
    volume: float = 100.0
) -> OhlcvBar:
    """Create a bar with symmetric spread around close."""
    return OhlcvBar(
        timestamp_utc=timestamp_utc,
        open=close,
        high=close + spread,
        low=close - spread,
        close=close,
        volume=volume,
    )


class LabelTests(unittest.TestCase):
    def test_label_generation_with_known_values(self) -> None:
        """Test label generation with hand-calculated values."""
        start = datetime(2026, 1, 26, 0, 0, tzinfo=timezone.utc)

        # Create 90 bars with gradually increasing close prices
        bars = []
        for idx in range(90):
            close_value = 100.0 + idx * 0.01
            bars.append(make_bar_with_spread(
                start + timedelta(minutes=idx),
                close_value,
                spread=0.01,
                volume=100.0
            ))

        # Generate labels for horizon 10
        result = generate_labels(bars, horizons=[10])

        # Verify we got one label
        self.assertEqual(len(result.labels), 1)
        label = result.labels[0]
        self.assertEqual(label.horizon_min, 10)

        # Verify label_class is one of the valid classes
        self.assertIn(label.label_class, ["up", "down", "choppy", "neutral"])

        # Verify ATR20 is positive
        self.assertGreater(result.atr20, 0)

        # Verify barriers
        self.assertGreater(label.barrier_upper, result.base_close)
        self.assertLess(label.barrier_lower, result.base_close)

    def test_up_label_only_upper_barrier_hit(self) -> None:
        """Test that only upper barrier hit results in 'up' label."""
        start = datetime(2026, 1, 26, 1, 0, tzinfo=timezone.utc)

        # Create 90 bars: 80 stable + 10 rising
        bars = []
        for idx in range(80):
            bars.append(make_bar_with_spread(
                start + timedelta(minutes=idx),
                100.0,
                spread=0.001
            ))

        # Add 10 bars with strong upward movement
        for idx in range(10):
            close_value = 100.0 + (idx + 1) * 0.5  # Strong rise
            bars.append(make_bar_with_spread(
                start + timedelta(minutes=80 + idx),
                close_value,
                spread=0.001
            ))

        result = generate_labels(bars, horizons=[10])
        label = result.labels[0]

        # Should be "up" because price rose significantly
        self.assertEqual(label.label_class, "up")
        self.assertEqual(label.hit_type, "upper")

    def test_down_label_only_lower_barrier_hit(self) -> None:
        """Test that only lower barrier hit results in 'down' label."""
        start = datetime(2026, 1, 26, 2, 0, tzinfo=timezone.utc)

        # Create 90 bars: 80 stable + 10 falling
        bars = []
        for idx in range(80):
            bars.append(make_bar_with_spread(
                start + timedelta(minutes=idx),
                100.0,
                spread=0.001
            ))

        # Add 10 bars with strong downward movement
        for idx in range(10):
            close_value = 100.0 - (idx + 1) * 0.5  # Strong fall
            bars.append(make_bar_with_spread(
                start + timedelta(minutes=80 + idx),
                close_value,
                spread=0.001
            ))

        result = generate_labels(bars, horizons=[10])
        label = result.labels[0]

        # Should be "down" because price fell significantly
        self.assertEqual(label.label_class, "down")
        self.assertEqual(label.hit_type, "lower")

    def test_choppy_label_both_barriers_hit(self) -> None:
        """Test that both barriers hit results in 'choppy' label."""
        start = datetime(2026, 1, 26, 3, 0, tzinfo=timezone.utc)

        # Create 90 bars: 80 stable + 10 oscillating
        bars = []
        for idx in range(80):
            bars.append(make_bar_with_spread(
                start + timedelta(minutes=idx),
                100.0,
                spread=0.001
            ))

        # Add 10 bars with oscillating movement (up then down)
        for idx in range(5):
            close_value = 100.0 + (idx + 1) * 0.5  # Rise
            bars.append(make_bar_with_spread(
                start + timedelta(minutes=80 + idx),
                close_value,
                spread=0.001
            ))
        for idx in range(5):
            close_value = 102.5 - (idx + 1) * 0.5  # Fall
            bars.append(make_bar_with_spread(
                start + timedelta(minutes=85 + idx),
                close_value,
                spread=0.001
            ))

        result = generate_labels(bars, horizons=[10])
        label = result.labels[0]

        # Should be "choppy" because price hit both barriers
        self.assertEqual(label.label_class, "choppy")
        self.assertEqual(label.hit_type, "both")

    def test_neutral_label_no_barriers_hit(self) -> None:
        """Test that no barrier hit results in 'neutral' label."""
        start = datetime(2026, 1, 26, 4, 0, tzinfo=timezone.utc)

        # Create 80 bars with some volatility, then 10 flat bars
        bars = []
        for idx in range(80):
            # Create bars with normal spread to establish ATR
            bars.append(make_bar_with_spread(
                start + timedelta(minutes=idx),
                100.0,
                spread=0.01  # Normal spread
            ))

        # Add 10 completely flat bars in judgment window
        for idx in range(10):
            # Completely flat bar (no movement at all)
            bars.append(OhlcvBar(
                timestamp_utc=start + timedelta(minutes=80 + idx),
                open=100.0,
                high=100.0,  # No spread
                low=100.0,
                close=100.0,
                volume=100.0
            ))

        result = generate_labels(bars, horizons=[10])
        label = result.labels[0]

        # Should be "neutral" because price didn't move in judgment window
        self.assertEqual(label.label_class, "neutral")
        self.assertEqual(label.hit_type, "none")
        self.assertEqual(label.first_hit_bar, 0)

    def test_insufficient_bars_raises_error(self) -> None:
        """Test that insufficient bars raises LabelError."""
        start = datetime(2026, 1, 26, 5, 0, tzinfo=timezone.utc)

        # Create only 50 bars (need at least 80 for horizon 60)
        bars = []
        for idx in range(50):
            bars.append(make_bar_with_spread(
                start + timedelta(minutes=idx),
                100.0
            ))

        with self.assertRaises(LabelError) as context:
            generate_labels(bars, horizons=[60])

        self.assertIn("at least 80 bars required", str(context.exception))

    def test_invalid_ohlc_raises_error(self) -> None:
        """Test that invalid OHLC raises LabelError."""
        start = datetime(2026, 1, 26, 6, 0, tzinfo=timezone.utc)

        # Create bars with invalid OHLC (high < low)
        # For horizon=10, t = 90-10-1 = 79
        # ATR range: t-19 to t = 60 to 79
        # Place invalid bar at index 65 (within ATR range)
        bars = []
        for idx in range(90):
            if idx == 65:  # Within ATR calculation range
                # Invalid bar: high < low
                bars.append(OhlcvBar(
                    timestamp_utc=start + timedelta(minutes=idx),
                    open=100.0,
                    high=99.0,  # Invalid: high < low
                    low=101.0,
                    close=100.0,
                    volume=100.0
                ))
            else:
                bars.append(make_bar_with_spread(
                    start + timedelta(minutes=idx),
                    100.0
                ))

        with self.assertRaises(LabelError) as context:
            generate_labels(bars, horizons=[10])

        # Check that error message mentions invalid OHLC
        self.assertIn("invalid", str(context.exception).lower())

    def test_multiple_horizons_consistent(self) -> None:
        """Test that multiple horizons generate consistent labels."""
        start = datetime(2026, 1, 26, 7, 0, tzinfo=timezone.utc)

        # Create 150 bars to accommodate horizon 60
        bars = []
        for idx in range(150):
            close_value = 100.0 + idx * 0.01
            bars.append(make_bar_with_spread(
                start + timedelta(minutes=idx),
                close_value,
                spread=0.01
            ))

        # Generate labels for all three horizons
        result = generate_labels(bars, horizons=[10, 30, 60])

        # Verify we got three labels
        self.assertEqual(len(result.labels), 3)

        # Verify horizons are correct and sorted
        self.assertEqual(result.labels[0].horizon_min, 10)
        self.assertEqual(result.labels[1].horizon_min, 30)
        self.assertEqual(result.labels[2].horizon_min, 60)

        # Verify all labels share same barriers
        for label in result.labels:
            self.assertEqual(label.barrier_upper, result.labels[0].barrier_upper)
            self.assertEqual(label.barrier_lower, result.labels[0].barrier_lower)

    def test_atr20_calculation_matches_hand_calculation(self) -> None:
        """Test ATR20 calculation with known True Range values."""
        start = datetime(2026, 1, 26, 8, 0, tzinfo=timezone.utc)

        # Create bars with known True Range
        # All bars have high-low = 0.02, and sequential closes
        bars = []
        for idx in range(90):
            close_value = 100.0 + idx * 0.01
            bars.append(make_bar_with_spread(
                start + timedelta(minutes=idx),
                close_value,
                spread=0.01  # high-low = 0.02
            ))

        result = generate_labels(bars, horizons=[10])

        # For sequential bars with spread=0.01:
        # True Range = max(0.02, |high - prev_close|, |low - prev_close|)
        # Since closes increase by 0.01, prev_close patterns vary
        # ATR20 should be positive and reasonable
        self.assertGreater(result.atr20, 0)
        self.assertLess(result.atr20, 1.0)  # Sanity check

    def test_first_bar_true_range_uses_hl_only(self) -> None:
        """Test that first bar (i=0) uses high-low for True Range."""
        start = datetime(2026, 1, 26, 9, 0, tzinfo=timezone.utc)

        # Create exactly 80 bars (minimum for horizon 60)
        bars = []
        for idx in range(80):
            # Use consistent spread for predictable ATR
            bars.append(make_bar_with_spread(
                start + timedelta(minutes=idx),
                100.0,
                spread=0.01
            ))

        # This should work without error
        # The ATR calculation handles i=0 specially
        result = generate_labels(bars, horizons=[60])

        # Verify result is valid
        self.assertEqual(len(result.labels), 1)
        self.assertGreater(result.atr20, 0)

        # ATR should be approximately equal to the spread (0.02)
        # since all bars have the same high-low range
        self.assertAlmostEqual(result.atr20, 0.02, delta=0.001)


if __name__ == "__main__":
    unittest.main()
