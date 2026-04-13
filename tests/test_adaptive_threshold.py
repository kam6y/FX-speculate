"""model/adaptive_threshold.py のテスト。"""

import pytest
from model.adaptive_threshold import AdaptiveThreshold


class TestComputeScaler:
    def test_normal_vol(self):
        at = AdaptiveThreshold({"horizon_1": 0.003})
        assert at.compute_scaler(0.005, 0.005) == pytest.approx(1.0)

    def test_high_vol_clamped(self):
        at = AdaptiveThreshold({"horizon_1": 0.003})
        assert at.compute_scaler(0.010, 0.005) == pytest.approx(1.5)

    def test_low_vol_clamped(self):
        at = AdaptiveThreshold({"horizon_1": 0.003})
        assert at.compute_scaler(0.002, 0.005) == pytest.approx(0.7)

    def test_moderate_high_vol(self):
        at = AdaptiveThreshold({"horizon_1": 0.003})
        assert at.compute_scaler(0.006, 0.005) == pytest.approx(1.2)


class TestGetThreshold:
    def test_h1_damped(self):
        at = AdaptiveThreshold(
            base_thresholds={"horizon_1": 0.003},
            damping_factor=0.5,
            damping_horizons=[1, 2, 3],
        )
        result = at.get_threshold(1, current_atr=0.0075, median_atr=0.005)
        assert result == pytest.approx(0.003 * 1.25)

    def test_h5_not_damped(self):
        at = AdaptiveThreshold(
            base_thresholds={"horizon_5": 0.003},
            damping_factor=0.5,
            damping_horizons=[1, 2, 3],
        )
        result = at.get_threshold(5, current_atr=0.0075, median_atr=0.005)
        assert result == pytest.approx(0.003 * 1.5)

    def test_normal_vol_no_change(self):
        at = AdaptiveThreshold(base_thresholds={"horizon_1": 0.003})
        result = at.get_threshold(1, current_atr=0.005, median_atr=0.005)
        assert result == pytest.approx(0.003)


class TestClassify:
    def test_up(self):
        at = AdaptiveThreshold(
            base_thresholds={"horizon_1": 0.003},
            abstain_margin=0.05,
        )
        assert at.classify(0.005, 1, 0.005, 0.005) == "UP"

    def test_down(self):
        at = AdaptiveThreshold(
            base_thresholds={"horizon_1": 0.003},
            abstain_margin=0.05,
        )
        assert at.classify(0.001, 1, 0.005, 0.005) == "DOWN"

    def test_abstain_near_threshold(self):
        at = AdaptiveThreshold(
            base_thresholds={"horizon_1": 0.003},
            abstain_margin=0.10,
        )
        assert at.classify(0.00305, 1, 0.005, 0.005) == "ABSTAIN"

    def test_just_outside_abstain_zone(self):
        at = AdaptiveThreshold(
            base_thresholds={"horizon_1": 0.003},
            abstain_margin=0.01,
        )
        assert at.classify(0.0031, 1, 0.005, 0.005) == "UP"
