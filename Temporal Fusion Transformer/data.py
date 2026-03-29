"""TFT データパイプライン

データ取得・特徴量エンジニアリング・前処理・TimeSeriesDataSet 作成を担当。
"""

import json
import os
import shutil
import tempfile
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# curl-cffi は非ASCII パスのCA証明書を読めないため、安全なパスにコピー
try:
    import certifi

    _cert = certifi.where()
    if not _cert.isascii():
        _safe = str(Path(tempfile.gettempdir()) / "cacert.pem")
        if not os.path.exists(_safe):
            shutil.copy2(_cert, _safe)
        os.environ["CURL_CA_BUNDLE"] = _safe
except (ImportError, OSError):
    pass

import numpy as np
import pandas as pd
import ta
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import EncoderNormalizer

from config import (
    CONFIG,
    ARTIFACT_DIR,
    FOMC_DATES,
    BOJ_DATES,
    KNOWN_CATEGORICALS,
    KNOWN_REALS,
    UNKNOWN_REALS_BASE,
    OPTIONAL_UNKNOWN,
    UNKNOWN_CATEGORICALS,
)

# ===================================================================
# オプション依存
# ===================================================================
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

try:
    from fredapi import Fred
    HAS_FRED = True
except ImportError:
    HAS_FRED = False

try:
    import holidays as _holidays_lib
    HAS_HOLIDAYS = True
except ImportError:
    HAS_HOLIDAYS = False


# ===================================================================
# データ取得
# ===================================================================
def fetch_market_data(config: dict) -> pd.DataFrame:
    """Yahoo Finance から USD/JPY, VIX, Gold, US10Y の日次データを取得"""
    if not HAS_YFINANCE:
        raise ImportError("yfinance が必要です: uv add yfinance")

    end = config["END_DATE"] or pd.Timestamp.now().strftime("%Y-%m-%d")
    start = config["START_DATE"]
    print(f"データ取得: {start} ~ {end}")

    tickers = ["JPY=X", "^VIX", "GC=F", "^TNX"]
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False, group_by="ticker")

    fx = raw["JPY=X"]
    df = pd.DataFrame({
        "open": fx["Open"].squeeze(),
        "high": fx["High"].squeeze(),
        "low": fx["Low"].squeeze(),
        "close": fx["Close"].squeeze(),
    })

    for ticker, col_name in [("^VIX", "vix"), ("GC=F", "gold"), ("^TNX", "us10y")]:
        data = raw[ticker]
        if not data["Close"].isna().all():
            df[col_name] = data["Close"].squeeze().reindex(df.index)

    if HAS_FRED and config.get("FRED_API_KEY"):
        try:
            fred = Fred(api_key=config["FRED_API_KEY"])
            jp10y = fred.get_series("IRLTLT01JPM156N", observation_start=start)
            jp10y_daily = jp10y.resample("D").ffill().reindex(df.index, method="ffill")
            if "us10y" in df.columns:
                df["interest_rate_diff"] = df["us10y"] - jp10y_daily
            print("  金利差: FRED API 使用")
        except Exception as e:
            print(f"  FRED API error: {e}")

    print(f"  取得行数: {len(df)}")
    return df


# ===================================================================
# 特徴量エンジニアリング
# ===================================================================
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """テクニカル指標を生成"""
    c, h, l = df["close"], df["high"], df["low"]

    df["log_return"] = np.log(c / c.shift(1))
    df["rsi_14"] = ta.momentum.RSIIndicator(c, window=14).rsi()

    macd_ind = ta.trend.MACD(c)
    df["macd"] = macd_ind.macd()
    df["macd_signal"] = macd_ind.macd_signal()
    df["macd_diff"] = macd_ind.macd_diff()

    bb = ta.volatility.BollingerBands(c, window=20, window_dev=2)
    df["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / c
    df["bb_pctb"] = bb.bollinger_pband()

    df["atr_14"] = ta.volatility.AverageTrueRange(h, l, c, window=14).average_true_range()

    for w in [5, 20, 50, 200]:
        sma = c.rolling(w).mean()
        df[f"ma_dist_{w}"] = (c - sma) / sma

    for s in [12, 26]:
        ema = c.ewm(span=s, adjust=False).mean()
        df[f"ema_dist_{s}"] = (c - ema) / ema

    return df


def _compute_event_distances(
    idx: pd.DatetimeIndex, event_dates: list[str], max_distance: float = 45.0
) -> np.ndarray:
    """各日付からイベント日への最小絶対距離 (O(n log m))"""
    events = np.sort(pd.to_datetime(event_dates).values)
    idx_arr = idx.values
    if len(events) == 0:
        return np.full(len(idx), max_distance)

    pos = np.searchsorted(events, idx_arr)
    one_day = np.timedelta64(1, "D")

    right_idx = np.minimum(pos, len(events) - 1)
    dist_right = np.abs((events[right_idx] - idx_arr) / one_day).astype(np.float64)
    dist_right = np.where(pos < len(events), dist_right, max_distance)

    left_idx = np.maximum(pos - 1, 0)
    dist_left = np.abs((idx_arr - events[left_idx]) / one_day).astype(np.float64)
    dist_left = np.where(pos > 0, dist_left, max_distance)

    return np.clip(np.minimum(dist_right, dist_left), 0, max_distance)


def _nfp_dates(start_year: int, end_year: int) -> list[str]:
    """米国雇用統計（NFP）発表日リスト"""
    dates: list[str] = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            d = date(year, month, 1)
            days_until_friday = (4 - d.weekday()) % 7
            first_friday = d + timedelta(days=days_until_friday)
            if first_friday.day == 1:
                first_friday += timedelta(days=7)
            dates.append(first_friday.isoformat())
    return dates


def add_event_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    """FOMC / BOJ / NFP イベント距離特徴量"""
    idx = df.index
    df["fomc_distance"] = _compute_event_distances(idx, FOMC_DATES)
    df["boj_distance"] = _compute_event_distances(idx, BOJ_DATES)
    nfp = _nfp_dates(idx[0].year, idx[-1].year + 1)
    df["nfp_distance"] = _compute_event_distances(idx, nfp)
    print(f"  イベント距離: FOMC({df['fomc_distance'].mean():.1f}d avg), "
          f"BOJ({df['boj_distance'].mean():.1f}d avg), "
          f"NFP({df['nfp_distance'].mean():.1f}d avg)")
    return df


def add_holiday_flags(df: pd.DataFrame) -> pd.DataFrame:
    """日米祝日フラグ"""
    if HAS_HOLIDAYS:
        years = range(df.index[0].year, df.index[-1].year + 2)
        us_hol = _holidays_lib.UnitedStates(years=years)
        jp_hol = _holidays_lib.Japan(years=years)
        df["holiday_us"] = df.index.isin(pd.to_datetime(list(us_hol.keys()))).astype(float)
        df["holiday_jp"] = df.index.isin(pd.to_datetime(list(jp_hol.keys()))).astype(float)
        print(f"  祝日フラグ: US={int(df['holiday_us'].sum())}日, JP={int(df['holiday_jp'].sum())}日")
    else:
        print("  holidays ライブラリ未インストール -フォールバック (全0)")
        df["holiday_us"] = 0.0
        df["holiday_jp"] = 0.0
    return df


def add_news_sentiment_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """ニュースセンチメント代理変数 (VIX ベース)"""
    if "vix" in df.columns and df["vix"].notna().sum() > 20:
        vix_ma = df["vix"].rolling(20, min_periods=1).mean()
        df["sentiment_proxy"] = -(df["vix"] - vix_ma) / vix_ma.clip(lower=1e-6)
        df["sentiment_proxy"] = df["sentiment_proxy"].clip(-1, 1)
        print(f"  センチメント代理: VIXベース (mean={df['sentiment_proxy'].mean():.3f})")
    else:
        df["sentiment_proxy"] = 0.0
        print("  センチメント代理: VIX 欠損のためゼロ埋め")
    return df


def add_market_regime(df: pd.DataFrame) -> pd.DataFrame:
    """市場レジームラベル (ADX + ATR ベース)"""
    c, h, l = df["close"], df["high"], df["low"]
    adx_val = ta.trend.ADXIndicator(h, l, c, window=14).adx()

    if "atr_14" in df.columns:
        atr_norm = df["atr_14"] / c
    else:
        atr_norm = ta.volatility.AverageTrueRange(h, l, c, window=14).average_true_range() / c

    vol_threshold = atr_norm.rolling(60, min_periods=1).quantile(0.75)
    adx_valid = adx_val.notna()
    regime = pd.Series("range", index=df.index)
    regime[adx_valid & (adx_val > 25)] = "trend"
    regime[adx_valid & (atr_norm > vol_threshold) & (adx_val <= 25)] = "high_vol"
    df["market_regime"] = regime.astype(str)

    counts = df["market_regime"].value_counts()
    print(f"  市場レジーム: {dict(counts)}")
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """カレンダー特徴量を生成"""
    idx = df.index
    df["day_of_week"] = idx.dayofweek.astype(str)
    df["month"] = idx.month.astype(str)
    df["quarter"] = idx.quarter.astype(str)

    df["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
    df["month_sin"] = np.sin(2 * np.pi * idx.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * idx.month / 12)
    df["doy_sin"] = np.sin(2 * np.pi * idx.dayofyear / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * idx.dayofyear / 365.25)

    df["is_month_start"] = idx.is_month_start.astype(float)
    df["is_month_end"] = idx.is_month_end.astype(float)
    df["is_quarter_end"] = idx.is_quarter_end.astype(float)
    return df


# ===================================================================
# 前処理
# ===================================================================
def preprocess(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """データ前処理: 欠損値補完 → 外れ値クリッピング → NaN除去"""
    cfg = config if config is not None else CONFIG
    df = df.ffill()

    if "log_return" in df.columns:
        train_end = int(len(df) * cfg["TRAIN_RATIO"])
        train_lr = df["log_return"].iloc[:train_end]
        mu, sigma = train_lr.mean(), train_lr.std()
        df["log_return"] = df["log_return"].clip(mu - 4 * sigma, mu + 4 * sigma)

    df = df.dropna()
    df = df[df.index.dayofweek < 5]
    df = df.copy()

    df["time_idx"] = np.arange(len(df))
    df["group_id"] = "USDJPY"

    print(f"前処理後: {len(df)} rows ({df.index[0].date()} ~ {df.index[-1].date()})")
    return df


def prepare_data(config: dict) -> pd.DataFrame:
    """データ取得 → 特徴量エンジニアリング → 前処理の一括パイプライン"""
    df = fetch_market_data(config)
    df = add_technical_indicators(df)
    df = add_calendar_features(df)
    df = add_event_distance_features(df)
    df = add_holiday_flags(df)
    df = add_news_sentiment_proxy(df)
    df = add_market_regime(df)
    return preprocess(df, config)


# ===================================================================
# 特徴量スキーマ
# ===================================================================
def build_unknown_reals(
    df: pd.DataFrame, config: dict | None = None, *, force_rebuild: bool = False
) -> list[str]:
    """利用可能な unknown reals を動的に構築し、スキーマを保存"""
    _SCHEMA_VERSION = 3
    schema_path = ARTIFACT_DIR / "feature_schema.json"

    if not force_rebuild and schema_path.exists():
        with open(schema_path) as f:
            saved = json.load(f)
        if isinstance(saved, dict) and saved.get("version") == _SCHEMA_VERSION:
            cols = saved["columns"]
            if all(c in df.columns for c in cols):
                return cols

    cfg = config if config is not None else CONFIG
    train_end = int(len(df) * cfg["TRAIN_RATIO"])
    df_train = df.iloc[:train_end]

    cols = list(UNKNOWN_REALS_BASE)
    for c in OPTIONAL_UNKNOWN:
        if c in df.columns and df_train[c].notna().sum() > len(df_train) * 0.5:
            cols.append(c)

    with open(schema_path, "w") as f:
        json.dump({"version": _SCHEMA_VERSION, "columns": cols}, f)

    return cols


def tsds_kwargs(
    config: dict,
    unknown_reals: list[str],
    unknown_cats: list[str],
    max_prediction_length: int | None = None,
) -> dict:
    """TimeSeriesDataSet 共通キーワード引数"""
    return dict(
        time_idx="time_idx",
        target="log_return",
        group_ids=["group_id"],
        max_encoder_length=config["MAX_ENCODER_LENGTH"],
        min_encoder_length=config["MIN_ENCODER_LENGTH"],
        max_prediction_length=max_prediction_length or config["MAX_PREDICTION_LENGTH"],
        static_categoricals=["group_id"],
        time_varying_known_categoricals=KNOWN_CATEGORICALS,
        time_varying_known_reals=KNOWN_REALS,
        time_varying_unknown_categoricals=unknown_cats,
        time_varying_unknown_reals=unknown_reals,
        target_normalizer=EncoderNormalizer(),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )


# ===================================================================
# データセット作成
# ===================================================================
def create_datasets(
    df: pd.DataFrame, config: dict
) -> tuple[TimeSeriesDataSet, TimeSeriesDataSet, TimeSeriesDataSet]:
    """Train / Validation / Test の TimeSeriesDataSet を作成"""
    n = len(df)
    train_end = int(n * config["TRAIN_RATIO"])
    val_end = int(n * (config["TRAIN_RATIO"] + config["VAL_RATIO"]))

    train_cutoff = int(df.iloc[train_end - 1]["time_idx"])
    val_cutoff = int(df.iloc[val_end - 1]["time_idx"])

    unknown_reals = build_unknown_reals(df)
    unknown_cats = [c for c in UNKNOWN_CATEGORICALS if c in df.columns]

    training = TimeSeriesDataSet(
        df[df["time_idx"] <= train_cutoff],
        **tsds_kwargs(config, unknown_reals, unknown_cats),
    )

    validation = TimeSeriesDataSet.from_dataset(
        training,
        df[df["time_idx"] <= val_cutoff],
        min_prediction_idx=train_cutoff + 1,
        stop_randomization=True,
    )

    test = TimeSeriesDataSet.from_dataset(
        training,
        df,
        min_prediction_idx=val_cutoff + 1,
        stop_randomization=True,
    )

    print(f"特徴量 -Known cat: {len(KNOWN_CATEGORICALS)}, Known real: {len(KNOWN_REALS)}, "
          f"Unknown cat: {len(unknown_cats)}, Unknown real: {len(unknown_reals)}")
    print(f"データセット -Train: {len(training)}, Val: {len(validation)}, Test: {len(test)}")

    return training, validation, test
