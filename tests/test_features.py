# tests/test_features.py
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from scripts.data_loader import load_data
from scripts.features import (
    prepare_ohlcv,
    generate_features,
    create_target,
    purged_time_series_split,
    purged_cv_splits,
    get_feature_columns,
    TIME_FEATURES,
)


@pytest.fixture
def sample_df():
    """テスト用に最初の1000行だけ読み込む"""
    df = load_data(str(Path(__file__).parent.parent / "data"))
    return df.head(1000)


def test_prepare_ohlcv(sample_df):
    """OHLCVカラムの準備"""
    df, price_cols = prepare_ohlcv(sample_df)
    assert "close" in price_cols
    assert "open" in price_cols
    assert "high" in price_cols
    assert "low" in price_cols
    assert "volume" in price_cols
    assert isinstance(df.index, pd.DatetimeIndex)


def test_generate_features(sample_df):
    """特徴量生成が100以上のカラムを生成する"""
    df, price_cols = prepare_ohlcv(sample_df)
    df_feat = generate_features(df, price_cols)
    assert len(df_feat.columns) > 100
    assert "ma_dist_20" in df_feat.columns
    assert "ema_dist_12" in df_feat.columns
    assert "atr_ratio" in df_feat.columns
    assert "bb_width" in df_feat.columns
    assert "return_1" in df_feat.columns
    assert "hour_sin" in df_feat.columns
    assert "is_tokyo_session" in df_feat.columns


def test_get_feature_columns(sample_df):
    """特徴量カラムのフィルタリング（price/target系を除外）"""
    df, price_cols = prepare_ohlcv(sample_df)
    df_feat = generate_features(df, price_cols)
    feat_cols = get_feature_columns(df_feat)
    for c in feat_cols:
        assert not c.startswith("ask_")
        assert not c.startswith("bid_")


def test_create_target():
    """ターゲットラベルの生成"""
    close = pd.Series([100.0, 100.1, 99.9, 100.05, 100.0],
                      index=pd.date_range("2024-01-01", periods=5, freq="min"))
    df = pd.DataFrame({"close": close})
    y = create_target(df, threshold_pips=5.0, horizon=1, pip_size=0.01)
    assert set(y.dropna().unique()).issubset({0, 1, 2})


def test_purged_time_series_split(sample_df):
    """Purged分割の比率とギャップ"""
    df, price_cols = prepare_ohlcv(sample_df)
    df_feat = generate_features(df, price_cols)
    train, val, test = purged_time_series_split(df_feat, train_ratio=0.6, val_ratio=0.2, gap_minutes=15)
    total = len(df_feat)
    assert len(train) == pytest.approx(total * 0.6, abs=2)
    assert len(val) < total * 0.2 + 1
    assert len(test) < total * 0.2 + 1
    gap = val.index[0] - train.index[-1]
    assert gap >= pd.Timedelta(minutes=15)


def test_time_features_constant():
    """TIME_FEATURESが9要素"""
    assert len(TIME_FEATURES) == 9
