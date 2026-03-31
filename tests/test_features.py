"""data/features.py のテスト。"""

import pandas as pd
import numpy as np
import pytest
from data.features import (
    apply_publication_lag,
    compute_technical_features,
    compute_market_returns,
    compute_calendar_features,
    build_features,
)


class TestApplyPublicationLag:
    def test_shifts_values_by_lag_days(self):
        index = pd.bdate_range("2025-01-02", periods=60)
        series = pd.Series(range(60), index=index)
        lagged = apply_publication_lag(series, lag_days=5)
        assert lagged.iloc[:5].isna().all()
        assert lagged.iloc[5] == series.iloc[0]

    def test_zero_lag_returns_original(self):
        index = pd.bdate_range("2025-01-02", periods=10)
        series = pd.Series(range(10), index=index)
        lagged = apply_publication_lag(series, lag_days=0)
        pd.testing.assert_series_equal(lagged, series)


class TestComputeTechnicalFeatures:
    def test_output_columns(self, sample_ohlcv):
        result = compute_technical_features(sample_ohlcv)
        expected = {"sma_5", "sma_20", "sma_60", "rsi_14", "macd", "bb_upper", "bb_lower", "atr"}
        assert expected.issubset(set(result.columns))

    def test_no_nan_after_warmup(self, sample_ohlcv):
        result = compute_technical_features(sample_ohlcv)
        assert result.iloc[60:].isna().sum().sum() == 0


class TestComputeMarketReturns:
    def test_output_columns(self, sample_ohlcv):
        df = pd.DataFrame({
            "usdjpy_close": sample_ohlcv["Close"],
            "sp500_close": sample_ohlcv["Close"] * 30,
            "nikkei_close": sample_ohlcv["Close"] * 250,
            "vix_close": 20 + np.random.randn(len(sample_ohlcv)),
            "oil_close": 70 + np.random.randn(len(sample_ohlcv)),
            "gold_close": 2000 + np.random.randn(len(sample_ohlcv)),
        }, index=sample_ohlcv.index)
        result = compute_market_returns(df)
        assert "log_return" in result.columns
        assert "sp500_return" in result.columns


class TestComputeCalendarFeatures:
    def test_output_columns(self):
        index = pd.bdate_range("2025-01-02", periods=30)
        result = compute_calendar_features(index)
        assert "day_of_week" in result.columns
        assert "month" in result.columns
        assert "is_month_end" in result.columns

    def test_day_of_week_range(self):
        index = pd.bdate_range("2025-01-02", periods=30)
        result = compute_calendar_features(index)
        assert result["day_of_week"].between(0, 4).all()


class TestBuildFeatures:
    def test_returns_complete_dataframe(self, sample_ohlcv):
        raw = pd.DataFrame({
            "usdjpy_open": sample_ohlcv["Open"],
            "usdjpy_high": sample_ohlcv["High"],
            "usdjpy_low": sample_ohlcv["Low"],
            "usdjpy_close": sample_ohlcv["Close"],
            "usdjpy_volume": sample_ohlcv["Volume"],
            "sp500_close": sample_ohlcv["Close"] * 30,
            "nikkei_close": sample_ohlcv["Close"] * 250,
            "vix_close": 20 + np.random.randn(len(sample_ohlcv)),
            "oil_close": 70 + np.random.randn(len(sample_ohlcv)),
            "gold_close": 2000 + np.random.randn(len(sample_ohlcv)),
            "fred_us_10y": np.full(len(sample_ohlcv), 4.5),
            "fred_jp_10y": np.full(len(sample_ohlcv), 1.0),
            "fred_ff_rate": np.full(len(sample_ohlcv), 5.25),
            "fred_cpi": np.full(len(sample_ohlcv), 310.0),
            "fred_unemployment": np.full(len(sample_ohlcv), 3.7),
            "fred_gdp": np.full(len(sample_ohlcv), 28000.0),
            "fred_m2": np.full(len(sample_ohlcv), 21000.0),
            "fred_dxy": np.full(len(sample_ohlcv), 104.0),
        }, index=sample_ohlcv.index)
        result = build_features(raw)
        assert "log_return" in result.columns
        assert len(result) > 0
        assert result.isna().sum().sum() == 0
