"""Script to split real OHLCV data and save to data/splits."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from pms.splitting import (
    split_timeseries,
    get_split_summary,
    get_split_bars,
    SplitConfig,
    SplitResult,
)
from pms.validation import OhlcvBar

DATA_FILE = ROOT.parent / "data" / "curated" / "ohlcv" / "ohlcv_usdjpy_1m_20240101_20260126.jsonl"
OUTPUT_DIR = ROOT.parent / "data" / "splits"

ISO8601_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def load_bars(filepath: Path, limit: int | None = None) -> list[OhlcvBar]:
    """Load OHLCV bars from JSONL file."""
    bars = []
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            record = json.loads(line)
            bar = OhlcvBar(
                timestamp_utc=datetime.strptime(
                    record["timestamp_utc"], ISO8601_FORMAT
                ).replace(tzinfo=timezone.utc),
                open=float(record["open"]),
                high=float(record["high"]),
                low=float(record["low"]),
                close=float(record["close"]),
                volume=float(record["volume"]),
            )
            bars.append(bar)
    return bars


def save_split_bars(
    bars: list[OhlcvBar],
    split_name: str,
    output_dir: Path,
    result: SplitResult,
) -> Path:
    """Save split bars to JSONL file and create manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename based on date range
    start_date = bars[0].timestamp_utc.strftime("%Y%m%d")
    end_date = bars[-1].timestamp_utc.strftime("%Y%m%d")
    filename = f"ohlcv_usdjpy_1m_{split_name}_{start_date}_{end_date}.jsonl"
    filepath = output_dir / filename

    # Write JSONL
    with open(filepath, "w", encoding="utf-8") as f:
        for bar in bars:
            record = {
                "timestamp_utc": bar.timestamp_utc.strftime(ISO8601_FORMAT),
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Write manifest
    manifest = {
        "dataset": f"ohlcv_{split_name}",
        "split": split_name,
        "file": filename,
        "symbol": "USDJPY",
        "timeframe_sec": 60,
        "records": len(bars),
        "start_timestamp_utc": bars[0].timestamp_utc.strftime(ISO8601_FORMAT),
        "end_timestamp_utc": bars[-1].timestamp_utc.strftime(ISO8601_FORMAT),
        "generated_at_utc": datetime.now(timezone.utc).strftime(ISO8601_FORMAT),
        "split_config": {
            "max_horizon_bars": result.config.max_horizon_bars,
            "min_test_days": result.config.min_test_days,
            "val_ratio": result.config.val_ratio,
            "purge_applied": True,
        },
        "schema": {
            "fields": ["timestamp_utc", "open", "high", "low", "close", "volume"],
            "order": "oldest_to_newest",
        },
    }
    manifest_path = output_dir / f"{filename.replace('.jsonl', '.manifest.json')}"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return filepath


def main():
    print(f"Loading data from: {DATA_FILE}")
    print("This may take a moment...")

    bars = load_bars(DATA_FILE)
    print(f"\nLoaded {len(bars):,} bars")
    print(f"Date range: {bars[0].timestamp_utc} ~ {bars[-1].timestamp_utc}")

    # Calculate total calendar days
    total_days = (bars[-1].timestamp_utc - bars[0].timestamp_utc).days
    print(f"Total calendar days: {total_days}")

    # Split with default config (90 days test period)
    print("\n" + "=" * 60)
    print("Splitting with default config (test period = 90 calendar days)")
    print("=" * 60)

    result = split_timeseries(bars)
    print("\n" + get_split_summary(result))

    # Show effective indices
    print("\nEffective indices (after purge):")
    train_start, train_end = result.get_train_indices()
    val_start, val_end = result.get_val_indices()
    test_start, test_end = result.get_test_indices()

    print(f"  Train: [{train_start:,} : {train_end:,}] = {train_end - train_start:,} bars")
    print(f"  Val:   [{val_start:,} : {val_end:,}] = {val_end - val_start:,} bars")
    print(f"  Test:  [{test_start:,} : {test_end:,}] = {test_end - test_start:,} bars")

    # Show percentage
    total = len(bars)
    print("\nPercentage breakdown:")
    print(f"  Train: {(train_end - train_start) / total * 100:.1f}%")
    print(f"  Val:   {(val_end - val_start) / total * 100:.1f}%")
    print(f"  Test:  {(test_end - test_start) / total * 100:.1f}%")
    print(f"  Purge: {result.purge_bars / total * 100:.2f}%")

    # Save splits to data/splits
    print("\n" + "=" * 60)
    print(f"Saving splits to: {OUTPUT_DIR}")
    print("=" * 60)

    for split_name in ["train", "val", "test"]:
        split_bars = get_split_bars(bars, result, split_name, apply_purge=True)
        filepath = save_split_bars(split_bars, split_name, OUTPUT_DIR, result)
        print(f"  {split_name}: {filepath.name} ({len(split_bars):,} bars)")

    print("\nDone!")


if __name__ == "__main__":
    main()
