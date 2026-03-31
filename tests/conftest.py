"""共通テストフィクスチャ。"""

import pandas as pd
import numpy as np
import pytest


@pytest.fixture
def sample_dates():
    """テスト用の営業日インデックス(250日 = 約1年)。"""
    return pd.bdate_range("2024-01-02", periods=250, freq="B")


@pytest.fixture
def sample_ohlcv(sample_dates):
    """テスト用の OHLCV データ。"""
    np.random.seed(42)
    n = len(sample_dates)
    close = 150.0 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame(
        {
            "Open": close + np.random.randn(n) * 0.1,
            "High": close + abs(np.random.randn(n) * 0.3),
            "Low": close - abs(np.random.randn(n) * 0.3),
            "Close": close,
            "Volume": np.random.randint(1000, 10000, n),
        },
        index=sample_dates,
    )
