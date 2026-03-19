"""共通特徴量生成モジュール

v6ノートブックの特徴量生成・ターゲット作成・データ分割ロジックを共通化。
"""

import numpy as np
import pandas as pd
import ta


TIME_FEATURES = [
    "hour_sin", "hour_cos", "minute_sin", "minute_cos",
    "day_sin", "day_cos",
    "is_tokyo_session", "is_london_session", "is_ny_session",
]


def prepare_ohlcv(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Bid/Ask parquetからOHLCV形式に変換。
    close = (ask_close + bid_close) / 2 (midprice)
    volume = 1.0 (FXデータのためダミー)
    indexをtimestampのDatetimeIndexに設定。

    NOTE: 元のask_*/bid_*カラムはそのまま保持される。
    run_backtest()がBid/Ask価格でエントリー/エグジットを計算するために必要。

    Returns: (df with datetime index, price_cols dict)
    """
    df = df.copy()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
        df.index.name = None

    if "ask_close" in df.columns and "bid_close" in df.columns:
        df["close"] = (df["ask_close"] + df["bid_close"]) / 2
        df["open"] = (df["ask_open"] + df["bid_open"]) / 2
        df["high"] = (df["ask_high"] + df["bid_high"]) / 2
        df["low"] = (df["ask_low"] + df["bid_low"]) / 2
    elif "close" not in df.columns:
        raise ValueError("DataFrameにclose価格カラムが見つかりません")

    if "volume" not in df.columns:
        df["volume"] = 1.0

    price_cols = {
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    }
    return df, price_cols


def generate_features(df: pd.DataFrame, price_cols: dict) -> pd.DataFrame:
    """全特徴量を生成して返す。

    1. ta.add_all_ta_features() によるテクニカル指標
    2. カスタム特徴量: MA乖離率, EMA乖離率, ATR比率, BB幅
    3. リターン特徴量: return_1, 5, 15, 60
    4. 時間特徴量: sin/cos, セッションフラグ
    """
    calc_df = df.copy()
    close = calc_df[price_cols["close"]]

    calc_df = ta.add_all_ta_features(
        calc_df,
        open=price_cols["open"],
        high=price_cols["high"],
        low=price_cols["low"],
        close=price_cols["close"],
        volume=price_cols["volume"],
        fillna=True,
    )
    calc_df = calc_df.replace([np.inf, -np.inf], np.nan)

    for window in [20, 50, 100, 200]:
        sma = close.rolling(window).mean()
        calc_df[f"ma_dist_{window}"] = (close - sma) / sma

    for window in [12, 26]:
        ema = close.ewm(span=window, adjust=False).mean()
        calc_df[f"ema_dist_{window}"] = (close - ema) / ema

    if "volatility_atr" in calc_df.columns:
        atr_ma = calc_df["volatility_atr"].rolling(30).mean()
        calc_df["atr_ratio"] = calc_df["volatility_atr"] / atr_ma

    if "volatility_bbh" in calc_df.columns and "volatility_bbl" in calc_df.columns:
        calc_df["bb_width"] = (calc_df["volatility_bbh"] - calc_df["volatility_bbl"]) / close

    for lag in [1, 5, 15, 60]:
        calc_df[f"return_{lag}"] = close.pct_change(lag)

    idx = calc_df.index
    hour = idx.hour
    minute = idx.minute
    day = idx.dayofweek

    calc_df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    calc_df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    calc_df["minute_sin"] = np.sin(2 * np.pi * minute / 60)
    calc_df["minute_cos"] = np.cos(2 * np.pi * minute / 60)
    calc_df["day_sin"] = np.sin(2 * np.pi * day / 7)
    calc_df["day_cos"] = np.cos(2 * np.pi * day / 7)

    calc_df["is_tokyo_session"] = ((hour >= 0) & (hour < 9)).astype(int)
    calc_df["is_london_session"] = ((hour >= 8) & (hour < 17)).astype(int)
    calc_df["is_ny_session"] = ((hour >= 13) & (hour < 22)).astype(int)

    return calc_df


def get_feature_columns(df_feat: pd.DataFrame) -> list[str]:
    """特徴量として使えるカラム名のリストを返す。
    price系・target系・raw系カラムを除外。
    NOTE: "volume"を含むカラム(volume_adi, volume_obv等)も除外される。
    これはv6からの仕様。FXデータではvolumeがダミー値(1.0)のため、
    volume系テクニカル指標は無意味な値になるので除外が正しい。
    """
    exclude_keywords = [
        "ask_", "bid_", "timestamp", "future", "target",
        "tick_", "open", "high", "low", "close", "volume",
    ]
    numeric_cols = df_feat.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if not any(k in c for k in exclude_keywords)]


def create_target(
    df: pd.DataFrame,
    threshold_pips: float,
    horizon: int = 15,
    pip_size: float = 0.01,
) -> pd.Series:
    """Buy(1)/Hold(0)/Sell(2) のターゲットラベルを生成。"""
    close = df["close"] if "close" in df.columns else df.iloc[:, 0]
    future_returns_pips = (close.shift(-horizon) - close) / pip_size
    y = pd.Series(0, index=df.index)
    y[future_returns_pips > threshold_pips] = 1
    y[future_returns_pips < -threshold_pips] = 2
    return y


def purged_cv_splits(n_samples: int, n_splits: int, gap: int):
    """Purged Cross-Validation splits。train末尾からgap行を除去してリークを防止。"""
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, val_idx in tscv.split(np.arange(n_samples)):
        if gap > 0 and len(train_idx) > gap:
            train_idx = train_idx[:-gap]
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        yield train_idx, val_idx


def purged_time_series_split(
    df: pd.DataFrame,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    gap_minutes: int = 15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Purged Time Series Split (60/20/20)。
    train-val間とval-test間にgap_minutesのギャップを挿入。
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    gap = gap_minutes

    train = df.iloc[:train_end]
    val_start = min(train_end + gap, n)
    val = df.iloc[val_start:val_end]
    test_start = min(val_end + gap, n)
    test = df.iloc[test_start:]

    return train, val, test
