# tests/test_evaluation.py
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from scripts.evaluation import (
    predict_with_thresholds,
    calc_trade_pnl,
    calculate_sharpe_ratio,
    trade_penalty,
    score_trading,
    build_live_filter,
    run_backtest,
    compute_metrics,
    DEFAULT_CONFIG,
)


def test_predict_with_thresholds():
    probs = np.array([
        [0.2, 0.6, 0.2],
        [0.8, 0.1, 0.1],
        [0.2, 0.2, 0.6],
    ])
    preds = predict_with_thresholds(probs, threshold_buy=0.4, threshold_sell=0.4)
    assert preds[0] == 1
    assert preds[1] == 0
    assert preds[2] == 2


def test_calculate_sharpe_ratio():
    returns = np.array([1.0, 1.0, 1.0, 1.0])
    sharpe = calculate_sharpe_ratio(returns, bar_per_year=362880, trade_rate=0.01)
    assert sharpe > 0


def test_calculate_sharpe_ratio_empty():
    assert calculate_sharpe_ratio(np.array([]), 362880, 0.01) == 0.0


def test_trade_penalty():
    penalty = trade_penalty(100, 10000, 50, 50, DEFAULT_CONFIG)
    assert 0.0 <= penalty <= 1.0


def test_build_live_filter():
    idx = pd.date_range("2024-01-01 20:00", periods=5, freq="h", tz="UTC")
    df_features = pd.DataFrame({"volatility_atr": [0.05, 0.05, 0.05, 0.05, 0.05]}, index=idx)
    config = DEFAULT_CONFIG.copy()
    eligible = build_live_filter(idx, df_features, config, atr_threshold=0.02)
    assert not eligible.iloc[0]
    assert not eligible.iloc[1]
    assert not eligible.iloc[2]
    assert not eligible.iloc[3]
    assert eligible.iloc[4]


def test_default_config_cost_model():
    assert DEFAULT_CONFIG["SPREAD_PIPS"] == 0.2
    assert DEFAULT_CONFIG["SLIPPAGE_PIPS"] == 0.1
    assert DEFAULT_CONFIG["API_FEE_RATE"] == 0.00002
