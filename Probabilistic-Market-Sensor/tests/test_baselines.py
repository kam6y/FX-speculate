"""Tests for baseline prediction module."""

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from pms.baselines import (  # noqa: E402
    RETURN_THRESHOLD_PCT,
    BaselineConfig,
    BaselineError,
    DirectionPredictionResult,
    RangePredictionResult,
    TrainedRangeBaseline,
    compute_return_1,
    predict_baseline_a,
    predict_baseline_a_batch,
    predict_baseline_b,
    predict_baseline_b_from_bars,
    train_range_baseline,
    train_range_baselines,
    predict_range_baseline,
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


class BaselineConfigTests(unittest.TestCase):
    """Tests for BaselineConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = BaselineConfig()
        self.assertEqual(config.horizons, [10, 30, 60])
        self.assertEqual(config.return_threshold_pct, RETURN_THRESHOLD_PCT)
        self.assertEqual(config.quantiles, [0.10, 0.50, 0.90])

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = BaselineConfig(
            horizons=[15, 45],
            return_threshold_pct=0.001,
            quantiles=[0.05, 0.50, 0.95],
        )
        self.assertEqual(config.horizons, [15, 45])
        self.assertEqual(config.return_threshold_pct, 0.001)
        self.assertEqual(config.quantiles, [0.05, 0.50, 0.95])

    def test_empty_horizons_raises(self) -> None:
        """Test that empty horizons raises error."""
        with self.assertRaises(BaselineError) as context:
            BaselineConfig(horizons=[])
        self.assertIn("horizons must not be empty", str(context.exception))

    def test_negative_horizon_raises(self) -> None:
        """Test that negative horizon raises error."""
        with self.assertRaises(BaselineError) as context:
            BaselineConfig(horizons=[10, -5, 60])
        self.assertIn("horizon must be > 0", str(context.exception))

    def test_negative_threshold_raises(self) -> None:
        """Test that negative threshold raises error."""
        with self.assertRaises(BaselineError) as context:
            BaselineConfig(return_threshold_pct=-0.01)
        self.assertIn("return_threshold_pct must be >= 0", str(context.exception))

    def test_wrong_quantiles_count_raises(self) -> None:
        """Test that wrong number of quantiles raises error."""
        with self.assertRaises(BaselineError) as context:
            BaselineConfig(quantiles=[0.10, 0.90])
        self.assertIn("quantiles must have exactly 3", str(context.exception))

    def test_unsorted_quantiles_raises(self) -> None:
        """Test that unsorted quantiles raises error."""
        with self.assertRaises(BaselineError) as context:
            BaselineConfig(quantiles=[0.90, 0.50, 0.10])
        self.assertIn("quantiles must be sorted ascending", str(context.exception))


class BaselineATests(unittest.TestCase):
    """Tests for Baseline-A (always neutral)."""

    def test_baseline_a_returns_neutral(self) -> None:
        """Test that Baseline-A always predicts neutral."""
        result = predict_baseline_a(30)
        self.assertEqual(result.predicted_class, "neutral")
        self.assertEqual(result.horizon_min, 30)

    def test_baseline_a_all_horizons(self) -> None:
        """Test Baseline-A for all standard horizons."""
        for horizon in [10, 30, 60]:
            result = predict_baseline_a(horizon)
            self.assertEqual(result.predicted_class, "neutral")
            self.assertEqual(result.horizon_min, horizon)

    def test_baseline_a_probabilities_sum_to_one(self) -> None:
        """Test that probabilities sum to 1.0."""
        result = predict_baseline_a(10)
        prob_sum = sum(result.probabilities.values())
        self.assertAlmostEqual(prob_sum, 1.0, places=6)

    def test_baseline_a_neutral_probability_is_one(self) -> None:
        """Test that neutral probability is 1.0 and others are 0.0."""
        result = predict_baseline_a(60)
        self.assertEqual(result.probabilities["neutral"], 1.0)
        self.assertEqual(result.probabilities["up"], 0.0)
        self.assertEqual(result.probabilities["down"], 0.0)
        self.assertEqual(result.probabilities["choppy"], 0.0)

    def test_baseline_a_batch(self) -> None:
        """Test batch prediction for Baseline-A."""
        results = predict_baseline_a_batch(n_samples=100, horizons=[10, 30, 60])
        self.assertEqual(len(results), 100)
        for sample_results in results:
            self.assertEqual(len(sample_results), 3)
            # Horizons should be sorted
            self.assertEqual(
                [r.horizon_min for r in sample_results],
                [10, 30, 60],
            )
            for r in sample_results:
                self.assertEqual(r.predicted_class, "neutral")

    def test_baseline_a_batch_empty(self) -> None:
        """Test batch prediction with zero samples."""
        results = predict_baseline_a_batch(n_samples=0, horizons=[30])
        self.assertEqual(len(results), 0)


class BaselineBTests(unittest.TestCase):
    """Tests for Baseline-B (1-minute return sign)."""

    def test_baseline_b_positive_return_predicts_up(self) -> None:
        """Test that positive return (above threshold) predicts 'up'."""
        # return_1 = 0.001 = 0.1% > 0.02% threshold
        result = predict_baseline_b(return_1=0.001, horizon=30)
        self.assertEqual(result.predicted_class, "up")
        self.assertEqual(result.probabilities["up"], 1.0)

    def test_baseline_b_negative_return_predicts_down(self) -> None:
        """Test that negative return (below -threshold) predicts 'down'."""
        # return_1 = -0.001 = -0.1% < -0.02% threshold
        result = predict_baseline_b(return_1=-0.001, horizon=30)
        self.assertEqual(result.predicted_class, "down")
        self.assertEqual(result.probabilities["down"], 1.0)

    def test_baseline_b_small_positive_return_predicts_neutral(self) -> None:
        """Test that small positive return predicts 'neutral'."""
        # |return_1| = 0.0001 = 0.01% < 0.02% threshold
        result = predict_baseline_b(return_1=0.0001, horizon=30)
        self.assertEqual(result.predicted_class, "neutral")

    def test_baseline_b_small_negative_return_predicts_neutral(self) -> None:
        """Test that small negative return predicts 'neutral'."""
        # |return_1| = 0.0001 = 0.01% < 0.02% threshold
        result = predict_baseline_b(return_1=-0.0001, horizon=30)
        self.assertEqual(result.predicted_class, "neutral")

    def test_baseline_b_zero_return_predicts_neutral(self) -> None:
        """Test that zero return predicts 'neutral'."""
        result = predict_baseline_b(return_1=0.0, horizon=30)
        self.assertEqual(result.predicted_class, "neutral")

    def test_baseline_b_threshold_boundary_positive(self) -> None:
        """Test behavior at positive threshold boundary."""
        threshold = RETURN_THRESHOLD_PCT

        # Exactly at threshold should be up (not neutral)
        result = predict_baseline_b(return_1=threshold, horizon=30)
        self.assertEqual(result.predicted_class, "up")

        # Just below threshold should be neutral
        result = predict_baseline_b(return_1=threshold - 0.00001, horizon=30)
        self.assertEqual(result.predicted_class, "neutral")

    def test_baseline_b_threshold_boundary_negative(self) -> None:
        """Test behavior at negative threshold boundary."""
        threshold = RETURN_THRESHOLD_PCT

        # Exactly at -threshold should be down
        result = predict_baseline_b(return_1=-threshold, horizon=30)
        self.assertEqual(result.predicted_class, "down")

        # Just above -threshold should be neutral
        result = predict_baseline_b(return_1=-threshold + 0.00001, horizon=30)
        self.assertEqual(result.predicted_class, "neutral")

    def test_baseline_b_never_predicts_choppy(self) -> None:
        """Test that Baseline-B never predicts 'choppy'."""
        test_returns = [-0.01, -0.001, -0.0001, 0.0, 0.0001, 0.001, 0.01]
        for ret in test_returns:
            result = predict_baseline_b(return_1=ret, horizon=30)
            self.assertNotEqual(result.predicted_class, "choppy")
            self.assertEqual(result.probabilities["choppy"], 0.0)

    def test_baseline_b_probabilities_sum_to_one(self) -> None:
        """Test that probabilities sum to 1.0 for all cases."""
        test_returns = [-0.01, 0.0, 0.01]
        for ret in test_returns:
            result = predict_baseline_b(return_1=ret, horizon=30)
            prob_sum = sum(result.probabilities.values())
            self.assertAlmostEqual(prob_sum, 1.0, places=6)

    def test_baseline_b_from_bars_positive_return(self) -> None:
        """Test Baseline-B prediction from OHLCV bars with positive return."""
        start = datetime(2026, 1, 27, 0, 0, tzinfo=timezone.utc)

        # Create bars with positive return (150.0 -> 150.5 = ~0.33% return)
        bars = [
            make_bar(start, close=150.0),
            make_bar(start + timedelta(minutes=1), close=150.5),
        ]

        results = predict_baseline_b_from_bars(bars, horizons=[10, 30, 60])

        self.assertEqual(len(results), 3)
        for r in results:
            self.assertEqual(r.predicted_class, "up")

    def test_baseline_b_from_bars_negative_return(self) -> None:
        """Test Baseline-B prediction from OHLCV bars with negative return."""
        start = datetime(2026, 1, 27, 0, 0, tzinfo=timezone.utc)

        # Create bars with negative return (150.0 -> 149.5 = ~-0.33% return)
        bars = [
            make_bar(start, close=150.0),
            make_bar(start + timedelta(minutes=1), close=149.5),
        ]

        results = predict_baseline_b_from_bars(bars, horizons=[10, 30, 60])

        self.assertEqual(len(results), 3)
        for r in results:
            self.assertEqual(r.predicted_class, "down")

    def test_baseline_b_from_bars_small_return(self) -> None:
        """Test Baseline-B prediction from OHLCV bars with small return."""
        start = datetime(2026, 1, 27, 0, 0, tzinfo=timezone.utc)

        # Create bars with very small return (150.0 -> 150.01 = ~0.0067% return)
        bars = [
            make_bar(start, close=150.0),
            make_bar(start + timedelta(minutes=1), close=150.01),
        ]

        results = predict_baseline_b_from_bars(bars, horizons=[10, 30, 60])

        self.assertEqual(len(results), 3)
        for r in results:
            self.assertEqual(r.predicted_class, "neutral")

    def test_baseline_b_from_bars_insufficient_bars_raises(self) -> None:
        """Test that insufficient bars raises error."""
        start = datetime(2026, 1, 27, 0, 0, tzinfo=timezone.utc)
        bars = [make_bar(start, close=150.0)]  # Only 1 bar

        with self.assertRaises(BaselineError) as context:
            predict_baseline_b_from_bars(bars, horizons=[30])
        self.assertIn("at least 2 bars", str(context.exception))

    def test_baseline_b_from_bars_horizons_sorted(self) -> None:
        """Test that results are sorted by horizon."""
        start = datetime(2026, 1, 27, 0, 0, tzinfo=timezone.utc)
        bars = [
            make_bar(start, close=150.0),
            make_bar(start + timedelta(minutes=1), close=150.5),
        ]

        # Pass unsorted horizons
        results = predict_baseline_b_from_bars(bars, horizons=[60, 10, 30])

        # Results should be sorted
        self.assertEqual([r.horizon_min for r in results], [10, 30, 60])


class ComputeReturn1Tests(unittest.TestCase):
    """Tests for compute_return_1 function."""

    def test_positive_return(self) -> None:
        """Test positive return calculation."""
        ret = compute_return_1(prev_close=100.0, close=101.0)
        self.assertAlmostEqual(ret, 0.01, places=6)  # 1%

    def test_negative_return(self) -> None:
        """Test negative return calculation."""
        ret = compute_return_1(prev_close=100.0, close=99.0)
        self.assertAlmostEqual(ret, -0.01, places=6)  # -1%

    def test_zero_return(self) -> None:
        """Test zero return calculation."""
        ret = compute_return_1(prev_close=100.0, close=100.0)
        self.assertAlmostEqual(ret, 0.0, places=6)

    def test_large_positive_return(self) -> None:
        """Test large positive return."""
        ret = compute_return_1(prev_close=100.0, close=150.0)
        self.assertAlmostEqual(ret, 0.50, places=6)  # 50%

    def test_large_negative_return(self) -> None:
        """Test large negative return."""
        ret = compute_return_1(prev_close=100.0, close=50.0)
        self.assertAlmostEqual(ret, -0.50, places=6)  # -50%

    def test_zero_prev_close_raises(self) -> None:
        """Test that zero prev_close raises error."""
        with self.assertRaises(BaselineError) as context:
            compute_return_1(prev_close=0.0, close=100.0)
        self.assertIn("prev_close must be > 0", str(context.exception))

    def test_negative_prev_close_raises(self) -> None:
        """Test that negative prev_close raises error."""
        with self.assertRaises(BaselineError) as context:
            compute_return_1(prev_close=-100.0, close=100.0)
        self.assertIn("prev_close must be > 0", str(context.exception))


class RangeBaselineTests(unittest.TestCase):
    """Tests for Range Baseline."""

    def test_train_range_baseline_computes_quantiles(self) -> None:
        """Test that training computes valid quantiles."""
        start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)

        # Create 100 bars with gradually increasing prices
        bars = [
            make_bar(start + timedelta(minutes=i), close=150.0 + i * 0.01)
            for i in range(100)
        ]

        trained = train_range_baseline(bars, horizon=10)

        self.assertEqual(trained.horizon_min, 10)
        # Quantiles should be ordered
        self.assertLessEqual(trained.return_q10, trained.return_q50)
        self.assertLessEqual(trained.return_q50, trained.return_q90)

    def test_train_range_baseline_uptrend(self) -> None:
        """Test range baseline on uptrend data."""
        start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)

        # Create uptrend: each bar increases by 0.1
        bars = [
            make_bar(start + timedelta(minutes=i), close=150.0 + i * 0.1)
            for i in range(100)
        ]

        trained = train_range_baseline(bars, horizon=10)

        # In uptrend, all quantiles should be positive
        self.assertGreater(trained.return_q10, 0)
        self.assertGreater(trained.return_q50, 0)
        self.assertGreater(trained.return_q90, 0)

    def test_train_range_baseline_stable_prices(self) -> None:
        """Test range baseline on stable prices."""
        start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)

        # Create stable prices (all same)
        bars = [
            make_bar(start + timedelta(minutes=i), close=150.0) for i in range(100)
        ]

        trained = train_range_baseline(bars, horizon=10)

        # All quantiles should be approximately 0
        self.assertAlmostEqual(trained.return_q10, 0.0, places=6)
        self.assertAlmostEqual(trained.return_q50, 0.0, places=6)
        self.assertAlmostEqual(trained.return_q90, 0.0, places=6)

    def test_train_range_baseline_insufficient_bars(self) -> None:
        """Test that insufficient bars raises error."""
        start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
        bars = [make_bar(start + timedelta(minutes=i), close=150.0) for i in range(5)]

        with self.assertRaises(BaselineError) as context:
            train_range_baseline(bars, horizon=10)
        self.assertIn("at least 11 bars", str(context.exception))

    def test_predict_range_baseline(self) -> None:
        """Test range baseline prediction."""
        start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
        bars = [
            make_bar(start + timedelta(minutes=i), close=150.0 + i * 0.01)
            for i in range(100)
        ]

        trained = train_range_baseline(bars, horizon=10)

        # Predict with current close = 151.0
        result = predict_range_baseline(151.0, trained)

        self.assertEqual(result.horizon_min, 10)
        # q10 <= q50 <= q90
        self.assertLessEqual(result.q10, result.q50)
        self.assertLessEqual(result.q50, result.q90)

    def test_predict_range_baseline_formula(self) -> None:
        """Test that prediction follows formula: close_t * (1 + q_p)."""
        trained = TrainedRangeBaseline(
            horizon_min=30,
            return_q10=-0.01,  # -1%
            return_q50=0.0,  # 0%
            return_q90=0.01,  # +1%
        )

        result = predict_range_baseline(100.0, trained)

        self.assertAlmostEqual(result.q10, 99.0, places=6)  # 100 * (1 - 0.01)
        self.assertAlmostEqual(result.q50, 100.0, places=6)  # 100 * (1 + 0)
        self.assertAlmostEqual(result.q90, 101.0, places=6)  # 100 * (1 + 0.01)

    def test_train_range_baselines_multiple_horizons(self) -> None:
        """Test training range baselines for multiple horizons."""
        start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
        bars = [
            make_bar(start + timedelta(minutes=i), close=150.0 + i * 0.01)
            for i in range(100)
        ]

        trained = train_range_baselines(bars, horizons=[10, 30, 60])

        self.assertIn(10, trained)
        self.assertIn(30, trained)
        self.assertIn(60, trained)

        for h, baseline in trained.items():
            self.assertEqual(baseline.horizon_min, h)

    def test_train_range_baselines_default_horizons(self) -> None:
        """Test training with default horizons."""
        start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
        bars = [
            make_bar(start + timedelta(minutes=i), close=150.0 + i * 0.01)
            for i in range(100)
        ]

        trained = train_range_baselines(bars)

        self.assertEqual(set(trained.keys()), {10, 30, 60})


class DirectionPredictionResultTests(unittest.TestCase):
    """Tests for DirectionPredictionResult dataclass."""

    def test_frozen_dataclass(self) -> None:
        """Test that DirectionPredictionResult is frozen."""
        result = DirectionPredictionResult(
            horizon_min=30,
            predicted_class="neutral",
            probabilities={"up": 0.0, "down": 0.0, "neutral": 1.0, "choppy": 0.0},
        )

        with self.assertRaises(AttributeError):
            result.horizon_min = 60  # type: ignore


class RangePredictionResultTests(unittest.TestCase):
    """Tests for RangePredictionResult dataclass."""

    def test_frozen_dataclass(self) -> None:
        """Test that RangePredictionResult is frozen."""
        result = RangePredictionResult(
            horizon_min=30,
            q10=149.0,
            q50=150.0,
            q90=151.0,
        )

        with self.assertRaises(AttributeError):
            result.q50 = 152.0  # type: ignore


if __name__ == "__main__":
    unittest.main()
