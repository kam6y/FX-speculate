"""Yahoo Finance / FRED からの生データ取得。

取得したデータは artifacts/raw_data.parquet にキャッシュされ、
当日中は再取得しない。
"""

import os
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from fredapi import Fred

from config import (
    DATA_YEARS,
    YAHOO_TICKERS,
    FRED_SERIES,
    PUBLICATION_LAGS,
    RAW_DATA_PATH,
    PROJECT_ROOT,
)

load_dotenv(PROJECT_ROOT / ".env")


def fetch_yahoo_data(ticker: str, years: int = DATA_YEARS) -> pd.DataFrame:
    """Yahoo Finance から日足データを取得する。"""
    end = date.today() + timedelta(days=1)  # yfinance の end は exclusive
    start = end - timedelta(days=years * 365)
    df = yf.download(ticker, start=str(start), end=str(end), progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()


def fetch_fred_data(series_id: str, years: int = DATA_YEARS, max_retries: int = 3) -> pd.Series:
    """FRED API から経済指標を取得する（リトライ付き）。"""
    api_key = os.environ.get("FRED_API_KEY", "")
    fred = Fred(api_key=api_key)
    end = date.today()
    start = end - timedelta(days=years * 365)
    for attempt in range(max_retries):
        try:
            series = fred.get_series(series_id, observation_start=str(start), observation_end=str(end))
            return series.dropna()
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 5 * (attempt + 1)
                print(f"  FRED retry {attempt+1}/{max_retries} for {series_id} (wait {wait}s): {e}")
                time.sleep(wait)
            else:
                raise


def fetch_all_data(years: int = DATA_YEARS, use_cache: bool = True) -> pd.DataFrame:
    """全データソースから取得し、USD/JPY の取引日ベースでマージする。"""
    # キャッシュチェック（日付 + 行数の両方を検証）
    min_expected_rows = years * 200  # 年間約200営業日
    if use_cache and RAW_DATA_PATH.exists():
        mtime = date.fromtimestamp(RAW_DATA_PATH.stat().st_mtime)
        if mtime == date.today():
            cached = pd.read_parquet(RAW_DATA_PATH)
            if len(cached) >= min_expected_rows:
                return cached
            print(f"  Cache has only {len(cached)} rows (expected >= {min_expected_rows}), refetching...")

    # USD/JPY をマスターカレンダーとして取得
    usdjpy = fetch_yahoo_data(YAHOO_TICKERS["usdjpy"], years)
    master_index = usdjpy.index

    merged = pd.DataFrame(index=master_index)

    # USD/JPY OHLCV
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        merged[f"usdjpy_{col.lower()}"] = usdjpy[col]

    # 他の Yahoo データ
    for name, ticker in YAHOO_TICKERS.items():
        if name == "usdjpy":
            continue
        df = fetch_yahoo_data(ticker, years)
        merged = pd.merge_asof(
            merged, df[["Close"]].rename(columns={"Close": f"{name}_close"}),
            left_index=True, right_index=True, direction="backward",
        )

    # FRED データ（公表ラグを考慮してインデックスをずらしてからマージ）
    for name, series_id in FRED_SERIES.items():
        series = fetch_fred_data(series_id, years)
        lag = PUBLICATION_LAGS.get(name, 0)
        # observation_date（参照期間終端日）→ 実際の公表日へインデックスをずらす
        shifted_index = series.index + pd.offsets.BusinessDay(lag)
        series_lagged = pd.Series(series.values, index=shifted_index, name=f"fred_{name}")
        fred_df = series_lagged.to_frame()
        merged = pd.merge_asof(
            merged, fred_df,
            left_index=True, right_index=True, direction="backward",
        )

    # forward fill で欠損を補完
    merged = merged.ffill().dropna()

    # キャッシュ保存
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(RAW_DATA_PATH)

    return merged
