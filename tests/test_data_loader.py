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


def test_load_data_returns_expected_columns(loaded_df):
    """load_dataが期待するカラムを含むDataFrameを返す"""
    expected_cols = [
        "timestamp", "ask_open", "ask_high", "ask_low", "ask_close",
        "bid_open", "bid_high", "bid_low", "bid_close"
    ]
    for col in expected_cols:
        assert col in loaded_df.columns, f"Missing column: {col}"


@pytest.fixture(scope="module")
def loaded_df():
    """テスト共有用のDataFrame（モジュール内で1回だけ読み込み）"""
    return load_data(str(Path(__file__).parent.parent / "data"))


def test_load_data_timestamp_is_utc(loaded_df):
    """timestampがUTC datetimeになっている"""
    assert pd.api.types.is_datetime64_any_dtype(loaded_df["timestamp"])
    assert str(loaded_df["timestamp"].dt.tz) == "UTC"


def test_load_data_no_duplicates(loaded_df):
    """重複タイムスタンプがない"""
    assert loaded_df["timestamp"].duplicated().sum() == 0


def test_load_data_raises_on_missing_dir():
    """存在しないディレクトリでFileNotFoundError"""
    with pytest.raises(FileNotFoundError):
        load_data("/nonexistent/path")
