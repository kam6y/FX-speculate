#!/usr/bin/env python3
"""Fetch USD/JPY 1m OHLC data from GMO Coin Forex public API.

Outputs normalized JSONL (UTC ISO8601, seconds=0) plus a manifest JSON.
Volume is set to 0.0 because the KLine endpoint does not provide volume.
MID is derived by averaging BID and ASK OHLC.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ISO8601_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
BASE_URL_DEFAULT = "https://forex-api.coin.z.com/public"
SYMBOL_DEFAULT = "USD_JPY"
OUTPUT_SYMBOL = "USDJPY"
INTERVAL_DEFAULT = "1min"
TIMEFRAME_SEC = 60


def parse_date(value: str) -> date:
    value = value.strip()
    if len(value) == 8 and value.isdigit():
        return datetime.strptime(value, "%Y%m%d").date()
    return datetime.strptime(value, "%Y-%m-%d").date()


def daterange(start: date, end: date) -> Iterable[date]:
    if end < start:
        raise ValueError("end-date must be >= start-date")
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def to_iso8601_from_ms(value: str) -> str:
    ts_ms = int(value)
    dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
    dt = dt.replace(second=0, microsecond=0)
    return dt.strftime(ISO8601_FORMAT)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def fetch_klines(
    base_url: str,
    symbol: str,
    price_type: str,
    interval: str,
    date_str: str,
    timeout_sec: float,
) -> List[Dict[str, str]]:
    params = {
        "symbol": symbol,
        "priceType": price_type,
        "interval": interval,
        "date": date_str,
    }
    url = f"{base_url}/v1/klines?{urlencode(params)}"
    request = Request(url, headers={"User-Agent": "pms-data-ingest"})
    with urlopen(request, timeout=timeout_sec) as response:
        payload = json.loads(response.read().decode("utf-8"))
    status = payload.get("status")
    if status != 0:
        raise RuntimeError(f"api status={status} for {date_str}: {payload}")
    return payload.get("data", [])


def build_kline_map(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    mapping: Dict[str, Dict[str, str]] = {}
    for row in rows:
        key = str(row["openTime"])
        if key not in mapping:
            mapping[key] = row
    return mapping


def midpoint(value_a: str, value_b: str) -> float:
    return (float(value_a) + float(value_b)) / 2.0


def write_jsonl(path: Path, records: List[Dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":"), ensure_ascii=True))
            handle.write("\n")


def write_json(path: Path, payload: Dict[str, object]) -> None:
    with path.open("w", newline="") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch GMO Coin FX KLine data")
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD or YYYYMMDD (JST)")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD or YYYYMMDD (JST)")
    parser.add_argument(
        "--price-type",
        required=True,
        choices=["BID", "ASK", "MID"],
        help="GMO priceType (BID/ASK) or MID (derived from BID+ASK)",
    )
    parser.add_argument("--output-dir", required=True, help="output directory for JSONL")
    parser.add_argument(
        "--manifest-dir",
        help="manifest output directory (default: same as output-dir)",
    )
    parser.add_argument("--base-url", default=BASE_URL_DEFAULT, help="API base URL")
    parser.add_argument("--symbol", default=SYMBOL_DEFAULT, help="GMO symbol (USD_JPY)")
    parser.add_argument("--interval", default=INTERVAL_DEFAULT, help="interval (default: 1min)")
    parser.add_argument("--sleep-sec", type=float, default=0.2, help="sleep between requests")
    parser.add_argument("--timeout-sec", type=float, default=20.0, help="HTTP timeout seconds")
    parser.add_argument("--source", default="GMO Coin Forex Public API", help="source label")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.symbol != SYMBOL_DEFAULT:
        raise ValueError("only USD_JPY is supported in this script")
    if args.interval != INTERVAL_DEFAULT:
        raise ValueError("only 1min interval is supported for v5.0 baseline")

    start = parse_date(args.start_date)
    end = parse_date(args.end_date)

    records_by_ts: Dict[str, Dict[str, object]] = {}
    total_days = 0
    missing_mid_days = 0

    for day in daterange(start, end):
        total_days += 1
        day_str = day.strftime("%Y%m%d")
        try:
            if args.price_type == "MID":
                bid_data = fetch_klines(
                    args.base_url,
                    args.symbol,
                    "BID",
                    args.interval,
                    day_str,
                    args.timeout_sec,
                )
                if args.sleep_sec > 0:
                    time.sleep(args.sleep_sec)
                ask_data = fetch_klines(
                    args.base_url,
                    args.symbol,
                    "ASK",
                    args.interval,
                    day_str,
                    args.timeout_sec,
                )
                bid_map = build_kline_map(bid_data)
                ask_map = build_kline_map(ask_data)
                shared_keys = sorted(set(bid_map.keys()) & set(ask_map.keys()))
                if not shared_keys:
                    missing_mid_days += 1
                for key in shared_keys:
                    bid_row = bid_map[key]
                    ask_row = ask_map[key]
                    timestamp_utc = to_iso8601_from_ms(key)
                    if timestamp_utc in records_by_ts:
                        continue
                    records_by_ts[timestamp_utc] = {
                        "timestamp_utc": timestamp_utc,
                        "open": midpoint(bid_row["open"], ask_row["open"]),
                        "high": midpoint(bid_row["high"], ask_row["high"]),
                        "low": midpoint(bid_row["low"], ask_row["low"]),
                        "close": midpoint(bid_row["close"], ask_row["close"]),
                        "volume": 0.0,
                    }
            else:
                data = fetch_klines(
                    args.base_url,
                    args.symbol,
                    args.price_type,
                    args.interval,
                    day_str,
                    args.timeout_sec,
                )
                for row in data:
                    timestamp_utc = to_iso8601_from_ms(row["openTime"])
                    if timestamp_utc in records_by_ts:
                        continue
                    records_by_ts[timestamp_utc] = {
                        "timestamp_utc": timestamp_utc,
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": 0.0,
                    }
        except (HTTPError, URLError, RuntimeError, ValueError) as exc:
            print(f"error: {day_str} fetch failed: {exc}", file=sys.stderr)
            return 1

        if args.sleep_sec > 0:
            time.sleep(args.sleep_sec)

    if not records_by_ts:
        print("error: no records fetched", file=sys.stderr)
        return 1

    records = sorted(records_by_ts.values(), key=lambda r: r["timestamp_utc"])
    start_ts = records[0]["timestamp_utc"]
    end_ts = records[-1]["timestamp_utc"]
    start_date = str(start_ts)[:10].replace("-", "")
    end_date = str(end_ts)[:10].replace("-", "")

    ensure_dir(Path(args.output_dir))
    output_name = f"ohlcv_usdjpy_1m_{start_date}_{end_date}.jsonl"
    output_path = Path(args.output_dir) / output_name
    write_jsonl(output_path, records)

    manifest_name = output_name.replace(".jsonl", ".manifest.json")
    manifest_dir = Path(args.manifest_dir) if args.manifest_dir else Path(args.output_dir)
    ensure_dir(manifest_dir)
    manifest_path = manifest_dir / manifest_name
    price_type_label = args.price_type
    if args.price_type == "MID":
        price_type_label = "MID(BID/ASK average)"
    source_detail = (
        f"{args.source} (symbol={args.symbol}, priceType={price_type_label}, interval={args.interval})"
    )
    manifest = {
        "dataset": "ohlcv",
        "symbol": OUTPUT_SYMBOL,
        "timeframe_sec": TIMEFRAME_SEC,
        "timezone": "UTC",
        "records": len(records),
        "start_timestamp_utc": start_ts,
        "end_timestamp_utc": end_ts,
        "source": source_detail,
        "generated_at_utc": datetime.now(timezone.utc).strftime(ISO8601_FORMAT),
        "file": output_name,
        "schema": {
            "order": "oldest_to_newest",
            "fields": ["timestamp_utc", "open", "high", "low", "close", "volume"],
        },
    }
    write_json(manifest_path, manifest)

    print(f"days fetched: {total_days}")
    if args.price_type == "MID" and missing_mid_days > 0:
        print(f"warning: {missing_mid_days} day(s) had no BID/ASK overlap", file=sys.stderr)
    print(f"records: {len(records)}")
    print(f"wrote {output_path}")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
