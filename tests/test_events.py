"""data/events.py のテスト。"""

import pandas as pd
import pytest
from data.events import (
    get_all_event_dates,
    compute_event_features,
)


class TestGetAllEventDates:
    def test_returns_sorted_dataframe(self):
        dates = get_all_event_dates(2025)
        assert isinstance(dates, pd.DataFrame)
        assert "date" in dates.columns
        assert "event_type" in dates.columns
        assert dates["date"].is_monotonic_increasing

    def test_contains_fomc(self):
        dates = get_all_event_dates(2025)
        fomc = dates[dates["event_type"] == "FOMC"]
        assert len(fomc) >= 8  # 年8回

    def test_contains_nfp(self):
        dates = get_all_event_dates(2025)
        nfp = dates[dates["event_type"] == "NFP"]
        assert len(nfp) == 12  # 月次

    def test_nfp_is_first_friday(self):
        dates = get_all_event_dates(2025)
        nfp = dates[dates["event_type"] == "NFP"]
        for _, row in nfp.iterrows():
            d = pd.Timestamp(row["date"])
            assert d.dayofweek == 4  # Friday
            assert d.day <= 7  # 第1週


class TestComputeEventFeatures:
    def test_output_columns(self):
        index = pd.bdate_range("2025-01-02", periods=60)
        features = compute_event_features(index)
        expected_cols = {
            "days_to_next_major_event",
            "days_from_last_major_event",
            "event_type_next",
            "is_event_day",
            "event_density_past_5d",
        }
        assert expected_cols.issubset(set(features.columns))

    def test_days_to_next_is_nonnegative(self):
        index = pd.bdate_range("2025-03-01", periods=30)
        features = compute_event_features(index)
        assert (features["days_to_next_major_event"] >= 0).all()

    def test_event_density_past_only(self):
        """event_density_past_5d は過去のみ参照し、未来情報を含まない。"""
        index = pd.bdate_range("2025-06-01", periods=30)
        features = compute_event_features(index)
        assert (features["event_density_past_5d"] >= 0).all()
        assert (features["event_density_past_5d"] <= 5).all()

    def test_event_density_excludes_current_and_future(self):
        """event_density_past_5d は当日・未来のイベントを含まない。"""
        index = pd.bdate_range("2025-01-02", periods=5)
        features = compute_event_features(index)

        jan2 = features.loc[pd.Timestamp("2025-01-02"), "event_density_past_5d"]
        # 2025-01-02 の過去5営業日に ISM(1/1) が含まれる → 1
        # だが NFP(1/3) は未来なので含まれない → 1であり2ではない
        assert jan2 == 1, f"1/2 の密度は ISM のみで 1 のはずだが {jan2}"

        jan6 = features.loc[pd.Timestamp("2025-01-06"), "event_density_past_5d"]
        # 2025-01-06 の過去5営業日に ISM(1/1) + NFP(1/3) が含まれる → 2
        assert jan6 == 2, f"1/6 には ISM+NFP で 2 のはずだが {jan6}"

        # 核心: 1/2 時点では 1/3 の NFP が含まれず、1/6 時点では含まれる
        assert jan6 > jan2, "未来のイベントが過去密度に漏れていないことの確認"

    def test_is_event_day_binary(self):
        index = pd.bdate_range("2025-01-02", periods=60)
        features = compute_event_features(index)
        assert set(features["is_event_day"].unique()).issubset({0, 1})
