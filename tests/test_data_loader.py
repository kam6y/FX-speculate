import pytest
import pandas as pd
from pathlib import Path
from scripts.data_loader import load_data, find_latest_parquet


def test_find_latest_parquet():
    """data/ディレクトリから最新のparquetファイルを検出できる"""
    path = find_latest_parquet(str(Path(__file__).parent.parent / "data"))
    assert path is not None
    assert path.suffix == ".parquet"
    assert path.name.startswith("usd_jpy_1min_")


def test_load_data_returns_expected_columns():
    """load_dataが期待するカラムを含むDataFrameを返す"""
    df = load_data(str(Path(__file__).parent.parent / "data"))
    expected_cols = [
        "timestamp", "ask_open", "ask_high", "ask_low", "ask_close",
        "bid_open", "bid_high", "bid_low", "bid_close"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing column: {col}"


def test_load_data_timestamp_is_utc():
    """timestampがUTC datetimeになっている"""
    df = load_data(str(Path(__file__).parent.parent / "data"))
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])


def test_load_data_no_duplicates():
    """重複タイムスタンプがない"""
    df = load_data(str(Path(__file__).parent.parent / "data"))
    assert df["timestamp"].duplicated().sum() == 0


def test_load_data_raises_on_missing_dir():
    """存在しないディレクトリでFileNotFoundError"""
    with pytest.raises(FileNotFoundError):
        load_data("/nonexistent/path")
