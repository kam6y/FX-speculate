"""共通データ読み込みモジュール"""

import re
from pathlib import Path
import pandas as pd


def find_latest_parquet(data_dir: str = "data") -> Path:
    """data_dir内の usd_jpy_1min_*.parquet から最新のファイルを検出。
    ファイル名末尾の日付(YYYYMMDD)をパースし、最も新しいものを返す。
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    pattern = re.compile(r"usd_jpy_1min_\d{8}_(\d{8})_utc\.parquet")
    candidates = []
    for f in data_path.glob("usd_jpy_1min_*.parquet"):
        m = pattern.match(f.name)
        if m:
            candidates.append((m.group(1), f))

    if not candidates:
        raise FileNotFoundError(f"No parquet files matching pattern in {data_dir}")

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def load_data(data_dir: str = "data") -> pd.DataFrame:
    """最新のparquetファイルを自動検出して読み込み。
    Returns: timestamp(UTC datetime), ask_open/high/low/close, bid_open/high/low/close
    """
    path = find_latest_parquet(data_dir)
    df = pd.read_parquet(path)

    # timestamp を datetime(UTC) に変換
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    elif df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")

    # 重複除去
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df
