"""model/confidence.py のテスト。"""

import numpy as np
import pytest
from model.confidence import ConfidenceEstimator


class TestEnsembleAgreement:
    def test_full_agreement_up(self):
        ce = ConfidenceEstimator()
        signals = [0.004, 0.005, 0.0031, 0.006, 0.0035]
        assert ce.ensemble_agreement(signals, threshold=0.003) == pytest.approx(1.0)

    def test_at_threshold_counted_as_down(self):
        ce = ConfidenceEstimator()
        signals = [0.003, 0.003, 0.003, 0.003, 0.003]
        assert ce.ensemble_agreement(signals, threshold=0.003) == pytest.approx(1.0)

    def test_full_agreement_down(self):
        ce = ConfidenceEstimator()
        signals = [0.001, 0.002, 0.0015, 0.0005, 0.0025]
        assert ce.ensemble_agreement(signals, threshold=0.003) == pytest.approx(1.0)

    def test_split_3_2(self):
        ce = ConfidenceEstimator()
        signals = [0.004, 0.005, 0.002, 0.001, 0.006]
        assert ce.ensemble_agreement(signals, threshold=0.003) == pytest.approx(0.6)

    def test_split_4_1(self):
        ce = ConfidenceEstimator()
        signals = [0.004, 0.005, 0.002, 0.006, 0.0035]
        assert ce.ensemble_agreement(signals, threshold=0.003) == pytest.approx(0.8)


class TestSpreadScore:
    def test_no_calibration_returns_half(self):
        ce = ConfidenceEstimator()
        assert ce.spread_score(0.01, -0.01) == pytest.approx(0.5)

    def test_smallest_spread_high_score(self):
        percentiles = np.array([0.005, 0.010, 0.015, 0.020])
        ce = ConfidenceEstimator(spread_percentiles=percentiles)
        assert ce.spread_score(0.002, -0.001) == pytest.approx(1.0)

    def test_largest_spread_low_score(self):
        percentiles = np.array([0.005, 0.010, 0.015, 0.020])
        ce = ConfidenceEstimator(spread_percentiles=percentiles)
        assert ce.spread_score(0.020, -0.010) == pytest.approx(0.0)

    def test_mid_spread(self):
        percentiles = np.array([0.005, 0.010, 0.015, 0.020])
        ce = ConfidenceEstimator(spread_percentiles=percentiles)
        assert ce.spread_score(0.006, -0.006) == pytest.approx(0.5)


class TestSignalStrength:
    def test_at_threshold(self):
        ce = ConfidenceEstimator()
        assert ce.signal_strength(0.003, 0.003) == pytest.approx(0.0)

    def test_far_from_threshold(self):
        ce = ConfidenceEstimator(signal_clip_max=2.0)
        assert ce.signal_strength(0.009, 0.003) == pytest.approx(1.0)

    def test_moderate_distance(self):
        ce = ConfidenceEstimator(signal_clip_max=2.0)
        assert ce.signal_strength(0.006, 0.003) == pytest.approx(0.5)


class TestCompositeScore:
    def test_all_high(self):
        ce = ConfidenceEstimator(
            weights=(0.4, 0.3, 0.3),
            signal_clip_max=2.0,
        )
        score = ce.score(
            per_model_signals=[0.005, 0.006, 0.004, 0.007, 0.005],
            threshold=0.003,
            q90=0.01,
            q10=-0.01,
            direction_signal=0.009,
        )
        assert score == pytest.approx(0.85)

    def test_weights_sum_to_one(self):
        ce = ConfidenceEstimator(weights=(0.5, 0.3, 0.2))
        assert sum(ce.weights) == pytest.approx(1.0)


class TestClassifyLevel:
    def test_high(self):
        ce = ConfidenceEstimator(confidence_boundaries=(0.4, 0.7))
        assert ce.classify_level(0.8) == "HIGH"

    def test_medium(self):
        ce = ConfidenceEstimator(confidence_boundaries=(0.4, 0.7))
        assert ce.classify_level(0.5) == "MEDIUM"

    def test_low(self):
        ce = ConfidenceEstimator(confidence_boundaries=(0.4, 0.7))
        assert ce.classify_level(0.3) == "LOW"

    def test_boundary_high(self):
        ce = ConfidenceEstimator(confidence_boundaries=(0.4, 0.7))
        assert ce.classify_level(0.7) == "HIGH"

    def test_boundary_medium(self):
        ce = ConfidenceEstimator(confidence_boundaries=(0.4, 0.7))
        assert ce.classify_level(0.4) == "MEDIUM"
