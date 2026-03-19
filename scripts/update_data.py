"""GMO Coin FX Public API からUSD/JPY 1分足データを取得し、既存parquetを更新する"""

import pandas as pd
import requests
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
EXISTING_FILE = DATA_DIR / "usd_jpy_1min_20231028_20260131_utc.parquet"
BASE_URL = "https://forex-api.coin.z.com/public/v1/klines"


def fetch_day(date_str: str, price_type: str, symbol: str = "USD_JPY") -> pd.DataFrame:
    """1日分のデータを取得する"""
    params = {
        "symbol": symbol,
        "interval": "1min",
        "date": date_str,
        "priceType": price_type,
    }
    resp = requests.get(BASE_URL, params=params, timeout=15)
    data = resp.json()

    if data.get("status") != 0:
        msg = data.get("messages", data.get("responsetime", ""))
        print(f"  WARN [{price_type}] {date_str}: status={data.get('status')} {msg}")
        return pd.DataFrame()

    rows = data.get("data", [])
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df


def parse_klines(df: pd.DataFrame) -> pd.DataFrame:
    """openTimeをUTC datetimeに変換しfloat化"""
    open_raw = pd.to_numeric(df["openTime"], errors="coerce")
    max_abs = open_raw.dropna().abs().max()
    if max_abs < 1e11:
        unit = "s"
    elif max_abs < 1e14:
        unit = "ms"
    elif max_abs < 1e17:
        unit = "us"
    else:
        unit = "ns"

    df["timestamp"] = pd.to_datetime(open_raw, unit=unit, utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])

    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].astype(float)

    return df


def fetch_range(start_date: str, end_date: str) -> pd.DataFrame:
    """start_date ~ end_date (YYYYMMDD) の範囲でASK/BIDデータを取得し結合"""
    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")

    all_ask = []
    all_bid = []
    current = start

    while current <= end:
        ds = current.strftime("%Y%m%d")
        print(f"Fetching {ds} ...", end=" ")

        df_ask = fetch_day(ds, "ASK")
        time.sleep(0.3)
        df_bid = fetch_day(ds, "BID")
        time.sleep(0.3)

        if len(df_ask) > 0:
            all_ask.append(df_ask)
        if len(df_bid) > 0:
            all_bid.append(df_bid)

        n_ask = len(df_ask)
        n_bid = len(df_bid)
        print(f"ASK={n_ask}, BID={n_bid}")

        current += timedelta(days=1)

    if not all_ask or not all_bid:
        raise ValueError("No data fetched")

    # ASKデータ
    ask_df = pd.concat(all_ask, ignore_index=True)
    ask_df = parse_klines(ask_df)
    ask_df = ask_df.rename(columns={
        "open": "ask_open",
        "high": "ask_high",
        "low": "ask_low",
        "close": "ask_close",
    })[["timestamp", "ask_open", "ask_high", "ask_low", "ask_close"]]

    # BIDデータ
    bid_df = pd.concat(all_bid, ignore_index=True)
    bid_df = parse_klines(bid_df)
    bid_df = bid_df.rename(columns={
        "open": "bid_open",
        "high": "bid_high",
        "low": "bid_low",
        "close": "bid_close",
    })[["timestamp", "bid_open", "bid_high", "bid_low", "bid_close"]]

    # ASK/BIDをtimestampでマージ
    merged = pd.merge(ask_df, bid_df, on="timestamp", how="inner")
    merged = merged.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    merged = merged.reset_index(drop=True)

    print(f"\nFetched total: {len(merged)} rows")
    print(f"  Range: {merged['timestamp'].min()} ~ {merged['timestamp'].max()}")
    return merged


def main():
    # 既存データ読み込み
    print(f"Loading existing data: {EXISTING_FILE}")
    existing = pd.read_parquet(EXISTING_FILE)
    last_ts = pd.Timestamp(existing["timestamp"].iloc[-1])
    print(f"Existing data: {len(existing)} rows, last={last_ts}")

    # 取得開始日: 既存データの最終日（JST基準で翌日から取得するため）
    # last_ts は UTC。APIの date は JST日付ベース
    # 安全のため、最終タイムスタンプの日(UTC)から取得し、重複はあとで除去
    start_date = last_ts.strftime("%Y%m%d")

    # 終了日: 今日(JST)
    jst_now = datetime.now(timezone.utc) + timedelta(hours=9)
    end_date = jst_now.strftime("%Y%m%d")

    print(f"Fetching: {start_date} ~ {end_date}")

    new_data = fetch_range(start_date, end_date)

    if len(new_data) == 0:
        print("No new data to append.")
        return

    # 既存データと同じカラム順序に揃える
    col_order = existing.columns.tolist()
    new_data = new_data[col_order]

    # 結合 & 重複除去
    combined = pd.concat([existing, new_data], ignore_index=True)
    combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
    combined = combined.sort_values("timestamp").reset_index(drop=True)

    added = len(combined) - len(existing)
    print(f"\nAdded {added} new rows")
    print(f"Total: {len(combined)} rows")
    print(f"Range: {combined['timestamp'].iloc[0]} ~ {combined['timestamp'].iloc[-1]}")

    # 新しいファイル名で保存
    last_new_ts = pd.Timestamp(combined["timestamp"].iloc[-1])
    new_end = last_new_ts.strftime("%Y%m%d")
    new_file = DATA_DIR / f"usd_jpy_1min_20231028_{new_end}_utc.parquet"

    combined.to_parquet(new_file, index=False)
    print(f"\nSaved: {new_file}")

    # 古いファイルとは別名で保存されるので、ノートブックのDATA_PATHを更新する必要がある
    print(f"\n*** NOTE: Update DATA_PATH in notebooks to: {new_file.name}")


if __name__ == "__main__":
    main()
