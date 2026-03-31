"""data/fetch.py のテスト。"""

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from data.fetch import fetch_yahoo_data, fetch_fred_data, fetch_all_data


class TestFetchYahooData:
    @patch("data.fetch.yf.download")
    def test_returns_dataframe_with_close(self, mock_download):
        mock_df = pd.DataFrame(
            {"Close": [150.0, 150.5]},
            index=pd.bdate_range("2025-01-02", periods=2),
        )
        mock_download.return_value = mock_df
        result = fetch_yahoo_data("USDJPY=X", years=1)
        assert isinstance(result, pd.DataFrame)
        assert "Close" in result.columns

    @patch("data.fetch.yf.download")
    def test_drops_na_rows(self, mock_download):
        mock_df = pd.DataFrame(
            {"Close": [150.0, None, 150.5]},
            index=pd.bdate_range("2025-01-02", periods=3),
        )
        mock_download.return_value = mock_df
        result = fetch_yahoo_data("USDJPY=X", years=1)
        assert result["Close"].isna().sum() == 0


class TestFetchFredData:
    @patch("data.fetch.Fred")
    def test_returns_series(self, mock_fred_cls):
        mock_fred = MagicMock()
        mock_fred.get_series.return_value = pd.Series(
            [5.0, 5.1], index=pd.to_datetime(["2025-01-01", "2025-02-01"])
        )
        mock_fred_cls.return_value = mock_fred
        result = fetch_fred_data("DGS10", years=1)
        assert isinstance(result, pd.Series)
        assert len(result) > 0


class TestFetchAllData:
    @patch("data.fetch.fetch_yahoo_data")
    @patch("data.fetch.fetch_fred_data")
    def test_returns_merged_dataframe(self, mock_fred, mock_yahoo):
        dates = pd.bdate_range("2025-01-02", periods=10)
        mock_yahoo.return_value = pd.DataFrame(
            {"Close": range(10), "Open": range(10),
             "High": range(10), "Low": range(10), "Volume": range(10)},
            index=dates,
        )
        mock_fred.return_value = pd.Series(range(10), index=dates)
        result = fetch_all_data(years=1, use_cache=False)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
