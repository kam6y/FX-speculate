"""特徴量エンジニアリング。

テクニカル指標、関連市場リターン、マクロ経済指標（公表ラグ適用済み）、
カレンダー特徴量、経済イベント特徴量を統合する。
"""

import numpy as np
import pandas as pd
import ta

from config import PUBLICATION_LAGS
from data.events import compute_event_features


def compute_technical_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """OHLCV からテクニカル指標を算出する。"""
    close = ohlcv.get("Close", ohlcv.get("usdjpy_close"))
    high = ohlcv.get("High", ohlcv.get("usdjpy_high"))
    low = ohlcv.get("Low", ohlcv.get("usdjpy_low"))

    result = pd.DataFrame(index=ohlcv.index)
    result["sma_5"] = ta.trend.sma_indicator(close, window=5)
    result["sma_20"] = ta.trend.sma_indicator(close, window=20)
    result["sma_60"] = ta.trend.sma_indicator(close, window=60)
    result["rsi_14"] = ta.momentum.rsi(close, window=14)
    result["macd"] = ta.trend.macd_diff(close)
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    result["bb_upper"] = bb.bollinger_hband()
    result["bb_lower"] = bb.bollinger_lband()
    result["atr"] = ta.volatility.average_true_range(high, low, close, window=14)
    return result


def compute_market_returns(df: pd.DataFrame) -> pd.DataFrame:
    """対数リターンを算出する。"""
    result = pd.DataFrame(index=df.index)
    result["log_return"] = np.log(df["usdjpy_close"] / df["usdjpy_close"].shift(1))
    result["log_return_5d"] = np.log(df["usdjpy_close"] / df["usdjpy_close"].shift(5))
    result["log_return_20d"] = np.log(df["usdjpy_close"] / df["usdjpy_close"].shift(20))
    for name in ["sp500", "nikkei", "vix", "oil", "gold"]:
        col = f"{name}_close"
        if col in df.columns:
            result[f"{name}_return"] = np.log(df[col] / df[col].shift(1))
    return result


def compute_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """マクロ経済指標の特徴量化（公表ラグは fetch.py 段階で適用済み）。"""
    result = pd.DataFrame(index=df.index)
    for name in PUBLICATION_LAGS.keys():
        col = f"fred_{name}"
        if col in df.columns:
            result[col] = df[col]
    if "fred_us_10y" in result.columns and "fred_jp_10y" in result.columns:
        result["rate_diff"] = result["fred_us_10y"] - result["fred_jp_10y"]
    return result


def compute_calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """カレンダー特徴量を算出する。"""
    result = pd.DataFrame(index=index)
    result["day_of_week"] = index.dayofweek
    result["month"] = index.month
    result["is_month_end"] = index.is_month_end.astype(int)
    return result


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """生データから全特徴量を構築する。"""
    ohlcv = raw[["usdjpy_open", "usdjpy_high", "usdjpy_low", "usdjpy_close", "usdjpy_volume"]]
    ohlcv_renamed = ohlcv.rename(columns={
        "usdjpy_open": "Open", "usdjpy_high": "High",
        "usdjpy_low": "Low", "usdjpy_close": "Close",
        "usdjpy_volume": "Volume",
    })
    technical = compute_technical_features(ohlcv_renamed)
    returns = compute_market_returns(raw)
    macro = compute_macro_features(raw)
    calendar = compute_calendar_features(raw.index)
    events = compute_event_features(raw.index)
    features = pd.concat([raw, technical, returns, macro, calendar, events], axis=1)
    features = features.loc[:, ~features.columns.duplicated()]
    features = features.dropna()
    return features
