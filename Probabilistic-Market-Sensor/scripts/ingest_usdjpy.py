#!/usr/bin/env python3
"""Data ingestion skeleton for USDJPY 1m OHLCV and macro events.

Normalizes timestamps to UTC ISO8601 with Z suffix and seconds=0,
orders records oldest->newest, and writes JSONL plus a manifest JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ISO8601_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
REQUIRED_SYMBOL = "USDJPY"
REQUIRED_TIMEFRAME_SEC = 60


def parse_iso8601(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_timestamp(value: str, fmt: str) -> datetime:
    if value is None or value == "":
        raise ValueError("timestamp is empty")
    if fmt == "iso8601":
        return parse_iso8601(str(value))
    if fmt == "epoch":
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if fmt == "epoch_ms":
        return datetime.fromtimestamp(float(value) / 1000.0, tz=timezone.utc)
    raise ValueError(f"unsupported timestamp format: {fmt}")


def normalize_minute(dt: datetime, allow_nonzero_seconds: bool) -> datetime:
    if dt.second != 0 or dt.microsecond != 0:
        if not allow_nonzero_seconds:
            raise ValueError("timestamp seconds or microseconds are not zero")
        dt = dt.replace(second=0, microsecond=0)
    return dt


def to_iso8601(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime(ISO8601_FORMAT)


def load_rows(path: Path, fmt: str) -> Iterable[Dict[str, Any]]:
    if fmt == "csv":
        with path.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                yield row
        return
    if fmt == "jsonl":
        with path.open("r") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
        return
    raise ValueError(f"unsupported input format: {fmt}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":"), ensure_ascii=True))
            handle.write("\n")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", newline="") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")


def infer_dates(records: List[Dict[str, Any]], key: str) -> Tuple[str, str]:
    start = records[0][key]
    end = records[-1][key]
    return start, end


def require_symbol(symbol: str) -> None:
    if symbol != REQUIRED_SYMBOL:
        raise ValueError(f"symbol must be {REQUIRED_SYMBOL}")


def require_timeframe_sec(timeframe_sec: int) -> None:
    if timeframe_sec != REQUIRED_TIMEFRAME_SEC:
        raise ValueError(f"timeframe_sec must be {REQUIRED_TIMEFRAME_SEC}")


def parse_ohlcv(args: argparse.Namespace) -> None:
    require_symbol(args.symbol)
    require_timeframe_sec(args.timeframe_sec)

    mapping = {
        "timestamp_utc": args.timestamp_col,
        "open": args.open_col,
        "high": args.high_col,
        "low": args.low_col,
        "close": args.close_col,
        "volume": args.volume_col,
    }

    records: List[Dict[str, Any]] = []
    seen_timestamps = set()

    for row in load_rows(Path(args.input), args.format):
        timestamp_raw = row.get(mapping["timestamp_utc"])
        if timestamp_raw in (None, ""):
            raise ValueError("missing timestamp_utc")
        dt = parse_timestamp(str(timestamp_raw), args.timestamp_format)
        dt = normalize_minute(dt, args.allow_nonzero_seconds)
        timestamp_utc = to_iso8601(dt)

        if timestamp_utc in seen_timestamps:
            raise ValueError(f"duplicate timestamp: {timestamp_utc}")
        seen_timestamps.add(timestamp_utc)

        open_value = row.get(mapping["open"])
        high_value = row.get(mapping["high"])
        low_value = row.get(mapping["low"])
        close_value = row.get(mapping["close"])
        volume_value = row.get(mapping["volume"])
        if open_value in (None, "") or high_value in (None, "") or low_value in (None, ""):
            raise ValueError(f"missing OHLC values at {timestamp_utc}")
        if close_value in (None, "") or volume_value in (None, ""):
            raise ValueError(f"missing close/volume at {timestamp_utc}")

        record = {
            "timestamp_utc": timestamp_utc,
            "open": float(open_value),
            "high": float(high_value),
            "low": float(low_value),
            "close": float(close_value),
            "volume": float(volume_value),
        }
        records.append(record)

    if not records:
        raise ValueError("no records loaded")

    records.sort(key=lambda r: r["timestamp_utc"])

    start_ts, end_ts = infer_dates(records, "timestamp_utc")
    start_date = start_ts[:10].replace("-", "")
    end_date = end_ts[:10].replace("-", "")

    ensure_dir(Path(args.output_dir))

    output_name = f"ohlcv_usdjpy_1m_{start_date}_{end_date}.jsonl"
    output_path = Path(args.output_dir) / output_name
    write_jsonl(output_path, records)

    manifest_name = output_name.replace(".jsonl", ".manifest.json")
    manifest_dir = Path(args.manifest_dir) if args.manifest_dir else Path(args.output_dir)
    ensure_dir(manifest_dir)
    manifest_path = manifest_dir / manifest_name
    manifest = {
        "dataset": "ohlcv",
        "symbol": args.symbol,
        "timeframe_sec": args.timeframe_sec,
        "timezone": "UTC",
        "records": len(records),
        "start_timestamp_utc": start_ts,
        "end_timestamp_utc": end_ts,
        "source": args.source,
        "generated_at_utc": to_iso8601(datetime.now(timezone.utc)),
        "file": output_name,
        "schema": {
            "order": "oldest_to_newest",
            "fields": ["timestamp_utc", "open", "high", "low", "close", "volume"],
        },
    }
    write_json(manifest_path, manifest)

    min_bars = args.min_coverage_days * 24 * 60
    if len(records) < min_bars:
        print(
            f"warning: only {len(records)} bars (< {min_bars} for {args.min_coverage_days} days)",
            file=sys.stderr,
        )

    print(f"wrote {len(records)} records -> {output_path}")
    print(f"wrote manifest -> {manifest_path}")


def parse_macro(args: argparse.Namespace) -> None:
    require_symbol(args.symbol)

    mapping = {
        "event_type": args.event_type_col,
        "scheduled_time_utc": args.scheduled_time_col,
        "importance": args.importance_col,
        "revision_policy": args.revision_policy_col,
        "published_at_utc": args.published_at_col,
        "actual": args.actual_col,
        "forecast": args.forecast_col,
        "previous": args.previous_col,
        "unit": args.unit_col,
    }

    allowlist = None
    if args.event_type_allowlist:
        allowlist_path = Path(args.event_type_allowlist)
        allowlist = {line.strip() for line in allowlist_path.read_text().splitlines() if line.strip()}

    records: List[Dict[str, Any]] = []
    seen_pairs = set()

    for row in load_rows(Path(args.input), args.format):
        event_type_raw = row.get(mapping["event_type"])
        if event_type_raw in (None, ""):
            raise ValueError("missing event_type")
        event_type = str(event_type_raw)
        if allowlist is not None and event_type not in allowlist:
            raise ValueError(f"event_type not in allowlist: {event_type}")

        scheduled_raw = row.get(mapping["scheduled_time_utc"])
        if scheduled_raw in (None, ""):
            raise ValueError(f"missing scheduled_time_utc for event_type {event_type}")
        scheduled_dt = parse_timestamp(str(scheduled_raw), args.timestamp_format)
        scheduled_dt = normalize_minute(scheduled_dt, args.allow_nonzero_seconds)
        scheduled_time_utc = to_iso8601(scheduled_dt)

        importance_raw = row.get(mapping["importance"])
        if importance_raw in (None, ""):
            raise ValueError(f"missing importance for event_type {event_type}")
        importance = str(importance_raw)
        importance = importance.lower()
        if importance not in {"low", "medium", "high"}:
            raise ValueError(f"invalid importance: {importance}")

        revision_raw = row.get(mapping["revision_policy"])
        if revision_raw in (None, ""):
            raise ValueError(f"missing revision_policy for event_type {event_type}")
        revision_policy = str(revision_raw)
        revision_policy = revision_policy.lower()
        if revision_policy not in {"none", "revision_possible", "revision_expected"}:
            raise ValueError(f"invalid revision_policy: {revision_policy}")

        published_at_value = row.get(mapping["published_at_utc"])
        published_at_utc = None
        if published_at_value not in (None, ""):
            published_dt = parse_timestamp(str(published_at_value), args.timestamp_format)
            published_dt = normalize_minute(published_dt, args.allow_nonzero_seconds)
            published_at_utc = to_iso8601(published_dt)

        pair_key = (event_type, scheduled_time_utc)
        if pair_key in seen_pairs:
            raise ValueError(f"duplicate event_type + scheduled_time_utc: {pair_key}")
        seen_pairs.add(pair_key)

        record: Dict[str, Any] = {
            "event_type": event_type,
            "scheduled_time_utc": scheduled_time_utc,
            "importance": importance,
            "revision_policy": revision_policy,
        }

        if published_at_utc is not None:
            record["published_at_utc"] = published_at_utc

        for key in ("actual", "forecast", "previous"):
            value = row.get(mapping[key])
            if value not in (None, ""):
                record[key] = float(value)

        unit_value = row.get(mapping["unit"])
        if unit_value not in (None, ""):
            record["unit"] = str(unit_value)

        records.append(record)

    if not records:
        raise ValueError("no records loaded")

    records.sort(key=lambda r: (r["scheduled_time_utc"], r["event_type"]))

    start_ts, end_ts = infer_dates(records, "scheduled_time_utc")
    start_date = start_ts[:10].replace("-", "")
    end_date = end_ts[:10].replace("-", "")

    ensure_dir(Path(args.output_dir))

    output_name = f"macro_events_{start_date}_{end_date}.jsonl"
    output_path = Path(args.output_dir) / output_name
    write_jsonl(output_path, records)

    manifest_name = output_name.replace(".jsonl", ".manifest.json")
    manifest_dir = Path(args.manifest_dir) if args.manifest_dir else Path(args.output_dir)
    ensure_dir(manifest_dir)
    manifest_path = manifest_dir / manifest_name
    manifest = {
        "dataset": "macro_events",
        "symbol": args.symbol,
        "timezone": "UTC",
        "records": len(records),
        "start_scheduled_time_utc": start_ts,
        "end_scheduled_time_utc": end_ts,
        "source": args.source,
        "generated_at_utc": to_iso8601(datetime.now(timezone.utc)),
        "file": output_name,
        "schema": {
            "order": "oldest_to_newest",
            "fields": [
                "event_type",
                "scheduled_time_utc",
                "importance",
                "revision_policy",
                "published_at_utc",
                "actual",
                "forecast",
                "previous",
                "unit",
            ],
        },
    }
    if allowlist is not None:
        manifest["event_type_allowlist"] = args.event_type_allowlist
    write_json(manifest_path, manifest)

    print(f"wrote {len(records)} records -> {output_path}")
    print(f"wrote manifest -> {manifest_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="USDJPY data ingestion skeleton")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--input", required=True, help="input file path")
    common.add_argument("--format", required=True, choices=["csv", "jsonl"], help="input format")
    common.add_argument("--output-dir", required=True, help="output directory")
    common.add_argument(
        "--manifest-dir",
        help="manifest output directory (default: same as output-dir)",
    )
    common.add_argument("--source", default="", help="data source label")
    common.add_argument("--symbol", default="USDJPY", help="symbol (default: USDJPY)")
    common.add_argument(
        "--timestamp-format",
        default="iso8601",
        choices=["iso8601", "epoch", "epoch_ms"],
        help="timestamp format",
    )
    common.add_argument(
        "--allow-nonzero-seconds",
        action="store_true",
        help="allow timestamps with non-zero seconds by truncating to minute",
    )

    ohlcv = subparsers.add_parser("ohlcv", parents=[common], help="ingest OHLCV")
    ohlcv.add_argument("--timeframe-sec", type=int, default=60, help="timeframe in seconds")
    ohlcv.add_argument("--timestamp-col", default="timestamp_utc", help="timestamp column")
    ohlcv.add_argument("--open-col", default="open", help="open column")
    ohlcv.add_argument("--high-col", default="high", help="high column")
    ohlcv.add_argument("--low-col", default="low", help="low column")
    ohlcv.add_argument("--close-col", default="close", help="close column")
    ohlcv.add_argument("--volume-col", default="volume", help="volume column")
    ohlcv.add_argument(
        "--min-coverage-days",
        type=int,
        default=90,
        help="minimum coverage days for warning (default: 90)",
    )
    ohlcv.set_defaults(func=parse_ohlcv)

    macro = subparsers.add_parser("macro", parents=[common], help="ingest macro events")
    macro.add_argument("--event-type-col", default="event_type", help="event_type column")
    macro.add_argument(
        "--scheduled-time-col",
        default="scheduled_time_utc",
        help="scheduled_time_utc column",
    )
    macro.add_argument("--importance-col", default="importance", help="importance column")
    macro.add_argument(
        "--revision-policy-col",
        default="revision_policy",
        help="revision_policy column",
    )
    macro.add_argument(
        "--published-at-col",
        default="published_at_utc",
        help="published_at_utc column",
    )
    macro.add_argument("--actual-col", default="actual", help="actual column")
    macro.add_argument("--forecast-col", default="forecast", help="forecast column")
    macro.add_argument("--previous-col", default="previous", help="previous column")
    macro.add_argument("--unit-col", default="unit", help="unit column")
    macro.add_argument(
        "--event-type-allowlist",
        help="path to allowlist file (one event_type per line)",
    )
    macro.set_defaults(func=parse_macro)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

