"""data/dataset.py のテスト。"""

import pandas as pd
import numpy as np
import pytest
from data.dataset import prepare_data, create_datasets, split_data


class TestPrepareData:
    def test_adds_time_idx_and_group(self):
        n = 200
        index = pd.bdate_range("2024-01-02", periods=n)
        df = pd.DataFrame({
            "log_return": np.random.randn(n) * 0.01,
            "sma_5": np.random.randn(n),
            "day_of_week": index.dayofweek,
            "month": index.month,
            "is_month_end": index.is_month_end.astype(int),
            "days_to_next_major_event": np.random.randint(0, 20, n),
            "event_type_next": np.random.choice(["FOMC", "NFP", "CPI"], n),
        }, index=index)
        result = prepare_data(df)
        assert "time_idx" in result.columns
        assert "group_id" in result.columns
        assert result["time_idx"].is_monotonic_increasing

    def test_no_nan_in_output(self):
        n = 200
        index = pd.bdate_range("2024-01-02", periods=n)
        df = pd.DataFrame({
            "log_return": np.random.randn(n) * 0.01,
            "sma_5": np.random.randn(n),
            "day_of_week": index.dayofweek,
            "month": index.month,
            "is_month_end": index.is_month_end.astype(int),
            "days_to_next_major_event": np.random.randint(0, 20, n),
            "event_type_next": np.random.choice(["FOMC", "NFP", "CPI"], n),
        }, index=index)
        result = prepare_data(df)
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        assert result[numeric_cols].isna().sum().sum() == 0


class TestSplitData:
    def test_split_sizes(self):
        n = 1000
        index = pd.bdate_range("2020-01-02", periods=n)
        df = pd.DataFrame({"time_idx": range(n), "group_id": "usdjpy"}, index=index)
        train, val, tune, test = split_data(df)
        assert len(train) + len(val) + len(tune) + len(test) == n
        assert len(train) > len(val)
        assert len(test) > len(tune)

    def test_temporal_order_preserved(self):
        n = 1000
        index = pd.bdate_range("2020-01-02", periods=n)
        df = pd.DataFrame({"time_idx": range(n), "group_id": "usdjpy"}, index=index)
        train, val, tune, test = split_data(df)
        assert train.index.max() < val.index.min()
        assert val.index.max() < tune.index.min()
        assert tune.index.max() < test.index.min()
