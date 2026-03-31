# TFT USD/JPY 予測AI 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** pytorch-forecasting の TFT を使い、USD/JPY の 5 日先マルチホライズン予測システムを構築する

**Architecture:** データ取得(Yahoo Finance/FRED) → 特徴量エンジニアリング → pytorch-forecasting TimeSeriesDataSet → TFT学習/評価 → Streamlit ダッシュボード + 日次自動予測。`config.py` で全パラメータ一元管理、1ファイル1責務のモジュール分割。

**Tech Stack:** Python 3.11+, uv, pytorch-forecasting, PyTorch Lightning, yfinance, fredapi, ta, Streamlit, Optuna, Plotly

**Design Spec:** `docs/superpowers/specs/2026-03-31-tft-usdjpy-prediction-design.md`

---

## ファイル構成

| ファイル | 責務 | タスク |
|---|---|---|
| `pyproject.toml` | プロジェクト定義・依存管理 | Task 1 |
| `config.py` | 全設定の一元管理 | Task 1 |
| `.gitignore` | gitignore 更新 | Task 1 |
| `data/__init__.py` | パッケージ | Task 1 |
| `data/events.py` | 経済イベントカレンダー | Task 2 |
| `data/fetch.py` | Yahoo Finance / FRED データ取得 | Task 3 |
| `data/features.py` | 特徴量エンジニアリング | Task 4 |
| `data/dataset.py` | TimeSeriesDataSet 構築 | Task 5 |
| `model/__init__.py` | パッケージ | Task 6 |
| `model/loss.py` | DirectionAwareQuantileLoss | Task 6 |
| `model/trainer.py` | Lightning Trainer 設定 | Task 7 |
| `scripts/train.py` | 学習エントリポイント | Task 8 |
| `scripts/evaluate.py` | 評価・可視化 | Task 9 |
| `scripts/predict.py` | 日次予測 | Task 10 |
| `dashboard/app.py` | Streamlit ダッシュボード | Task 11 |
| `scripts/schedule_task.bat` | Windows タスクスケジューラ | Task 12 |
| `tests/` | 各モジュールのテスト | 各タスク内 |

---

### Task 1: プロジェクトスキャフォールディング

**Files:**
- Create: `pyproject.toml`
- Create: `config.py`
- Modify: `.gitignore`
- Create: `data/__init__.py`, `model/__init__.py`, `scripts/__init__.py`, `tests/__init__.py`, `tests/conftest.py`

- [ ] **Step 1: pyproject.toml を作成**

```toml
[project]
name = "fx-speculate"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "pytorch-forecasting>=1.0",
    "lightning>=2.0",
    "torch>=2.0",
    "yfinance>=0.2",
    "fredapi>=0.5",
    "python-dotenv>=1.0",
    "ta>=0.11",
    "pandas>=2.0",
    "numpy>=1.24",
    "matplotlib>=3.7",
    "seaborn>=0.13",
    "optuna>=3.0",
    "streamlit>=1.30",
    "plotly>=5.18",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-cov>=4.0"]
```

- [ ] **Step 2: config.py を作成**

```python
"""全設定の一元管理。各モジュールはここから import する。"""

from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).parent
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)
RAW_DATA_PATH = ARTIFACT_DIR / "raw_data.parquet"
PREDICTIONS_DB = ARTIFACT_DIR / "predictions.db"
OPTUNA_DB = ARTIFACT_DIR / "optuna_study.db"

# --- Data ---
DATA_YEARS = 10
YAHOO_TICKERS = {
    "usdjpy": "USDJPY=X",
    "sp500": "^GSPC",
    "nikkei": "^N225",
    "vix": "^VIX",
    "oil": "CL=F",
    "gold": "GC=F",
}
FRED_SERIES = {
    "us_10y": "DGS10",
    "jp_10y": "IRLTLT01JPM156N",
    "ff_rate": "FEDFUNDS",
    "cpi": "CPIAUCSL",
    "unemployment": "UNRATE",
    "gdp": "GDP",
    "m2": "M2SL",
    "dxy": "DTWEXBGS",
}
PUBLICATION_LAGS = {
    "cpi": 35,
    "unemployment": 32,
    "gdp": 30,
    "m2": 14,
    "ff_rate": 1,
    "us_10y": 1,
    "jp_10y": 45,
    "dxy": 1,
}

# --- Model ---
ENCODER_LENGTH = 60
PREDICTION_LENGTH = 5
HIDDEN_SIZE = 64
ATTENTION_HEAD_SIZE = 4
DROPOUT = 0.2
HIDDEN_CONTINUOUS_SIZE = 32
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]
OUTPUT_SIZE = len(QUANTILES)

# --- Loss ---
DIRECTION_WEIGHT = 1.0
SMOOTHING_TEMPERATURE = 0.1
DEAD_ZONE = 1e-4

# --- Training ---
MAX_EPOCHS = 100
EARLY_STOP_PATIENCE = 10
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
TOP_K_CHECKPOINTS = 3
DATA_SPLIT_RATIOS = {
    "train": 0.65,
    "val": 0.15,
    "threshold_tune": 0.05,
    "test": 0.15,
}
```

- [ ] **Step 3: .gitignore を更新**

`.gitignore` に以下を追加:

```
.env
.vscode/
__pycache__
artifacts/
*.pyc
.pytest_cache/
lightning_logs/
```

- [ ] **Step 4: ディレクトリ・パッケージファイルを作成**

```bash
mkdir -p data model scripts tests dashboard
touch data/__init__.py model/__init__.py scripts/__init__.py tests/__init__.py
```

`tests/conftest.py`:

```python
"""共通テストフィクスチャ。"""

import pandas as pd
import numpy as np
import pytest


@pytest.fixture
def sample_dates():
    """テスト用の営業日インデックス(250日 = 約1年)。"""
    return pd.bdate_range("2024-01-02", periods=250, freq="B")


@pytest.fixture
def sample_ohlcv(sample_dates):
    """テスト用の OHLCV データ。"""
    np.random.seed(42)
    n = len(sample_dates)
    close = 150.0 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame(
        {
            "Open": close + np.random.randn(n) * 0.1,
            "High": close + abs(np.random.randn(n) * 0.3),
            "Low": close - abs(np.random.randn(n) * 0.3),
            "Close": close,
            "Volume": np.random.randint(1000, 10000, n),
        },
        index=sample_dates,
    )
```

- [ ] **Step 5: uv sync で依存パッケージをインストール**

```bash
uv sync
```

- [ ] **Step 6: テスト実行基盤を確認**

```bash
uv run pytest tests/ -v --co
```

Expected: テストが収集されること（0 tests collected でもエラーなし）

- [ ] **Step 7: コミット**

```bash
git add pyproject.toml config.py .gitignore data/ model/ scripts/ tests/
git commit -m "feat: プロジェクトスキャフォールディング（config, 依存関係, ディレクトリ構成）"
```

---

### Task 2: 経済イベントカレンダー (`data/events.py`)

**Files:**
- Create: `data/events.py`
- Create: `tests/test_events.py`

- [ ] **Step 1: tests/test_events.py にテストを作成**

```python
"""data/events.py のテスト。"""

import pandas as pd
import pytest
from data.events import (
    get_all_event_dates,
    compute_event_features,
)


class TestGetAllEventDates:
    def test_returns_sorted_dataframe(self):
        dates = get_all_event_dates(2025)
        assert isinstance(dates, pd.DataFrame)
        assert "date" in dates.columns
        assert "event_type" in dates.columns
        assert dates["date"].is_monotonic_increasing

    def test_contains_fomc(self):
        dates = get_all_event_dates(2025)
        fomc = dates[dates["event_type"] == "FOMC"]
        assert len(fomc) >= 8  # 年8回

    def test_contains_nfp(self):
        dates = get_all_event_dates(2025)
        nfp = dates[dates["event_type"] == "NFP"]
        assert len(nfp) == 12  # 月次

    def test_nfp_is_first_friday(self):
        dates = get_all_event_dates(2025)
        nfp = dates[dates["event_type"] == "NFP"]
        for _, row in nfp.iterrows():
            d = pd.Timestamp(row["date"])
            assert d.dayofweek == 4  # Friday
            assert d.day <= 7  # 第1週


class TestComputeEventFeatures:
    def test_output_columns(self):
        index = pd.bdate_range("2025-01-02", periods=60)
        features = compute_event_features(index)
        expected_cols = {
            "days_to_next_major_event",
            "days_from_last_major_event",
            "event_type_next",
            "is_event_day",
            "event_density_past_5d",
        }
        assert expected_cols.issubset(set(features.columns))

    def test_days_to_next_is_nonnegative(self):
        index = pd.bdate_range("2025-03-01", periods=30)
        features = compute_event_features(index)
        assert (features["days_to_next_major_event"] >= 0).all()

    def test_event_density_past_only(self):
        """event_density_past_5d は過去のみ参照し、未来情報を含まない。"""
        index = pd.bdate_range("2025-06-01", periods=30)
        features = compute_event_features(index)
        assert (features["event_density_past_5d"] >= 0).all()
        assert (features["event_density_past_5d"] <= 5).all()

    def test_is_event_day_binary(self):
        index = pd.bdate_range("2025-01-02", periods=60)
        features = compute_event_features(index)
        assert set(features["is_event_day"].unique()).issubset({0, 1})
```

- [ ] **Step 2: テスト実行、失敗を確認**

```bash
uv run pytest tests/test_events.py -v
```

Expected: ImportError (data.events が存在しない)

- [ ] **Step 3: data/events.py を実装**

```python
"""経済イベントカレンダー。

FOMC、日銀会合、米雇用統計(NFP)、CPI、GDP、ISM、小売売上、
ジャクソンホール会議の日程を管理し、イベント特徴量を算出する。
"""

import pandas as pd
import numpy as np

# --- 不規則イベント: 年次リストで手動管理 ---
FOMC_DATES = {
    2016: ["01-27", "03-16", "04-27", "06-15", "07-27", "09-21", "11-02", "12-14"],
    2017: ["02-01", "03-15", "05-03", "06-14", "07-26", "09-20", "11-01", "12-13"],
    2018: ["01-31", "03-21", "05-02", "06-13", "08-01", "09-26", "11-08", "12-19"],
    2019: ["01-30", "03-20", "05-01", "06-19", "07-31", "09-18", "10-30", "12-11"],
    2020: ["01-29", "03-03", "03-15", "04-29", "06-10", "07-29", "09-16", "11-05", "12-16"],
    2021: ["01-27", "03-17", "04-28", "06-16", "07-28", "09-22", "11-03", "12-15"],
    2022: ["01-26", "03-16", "05-04", "06-15", "07-27", "09-21", "11-02", "12-14"],
    2023: ["02-01", "03-22", "05-03", "06-14", "07-26", "09-20", "11-01", "12-13"],
    2024: ["01-31", "03-20", "05-01", "06-12", "07-31", "09-18", "11-07", "12-18"],
    2025: ["01-29", "03-19", "05-07", "06-18", "07-30", "09-17", "10-29", "12-17"],
    2026: ["01-28", "03-18", "04-29", "06-17", "07-29", "09-16", "11-04", "12-16"],
}

BOJ_DATES = {
    2016: ["01-29", "03-15", "04-28", "06-16", "07-29", "09-21", "11-01", "12-20"],
    2017: ["01-31", "03-16", "04-27", "06-16", "07-20", "09-21", "10-31", "12-21"],
    2018: ["01-23", "03-09", "04-27", "06-15", "07-31", "09-19", "10-31", "12-20"],
    2019: ["01-23", "03-15", "04-25", "06-20", "07-30", "09-19", "10-31", "12-19"],
    2020: ["01-21", "03-16", "04-27", "06-16", "07-15", "09-17", "10-29", "12-18"],
    2021: ["01-21", "03-19", "04-27", "06-18", "07-16", "09-22", "10-28", "12-17"],
    2022: ["01-18", "03-18", "04-28", "06-17", "07-21", "09-22", "10-28", "12-20"],
    2023: ["01-18", "03-10", "04-28", "06-16", "07-28", "09-22", "10-31", "12-19"],
    2024: ["01-23", "03-19", "04-26", "06-14", "07-31", "09-20", "10-31", "12-19"],
    2025: ["01-24", "03-14", "05-01", "06-17", "07-31", "09-19", "10-30", "12-19"],
    2026: ["01-22", "03-13", "04-28", "06-16", "07-16", "09-17", "10-29", "12-18"],
}

JACKSON_HOLE = {
    2016: ["08-26"], 2017: ["08-25"], 2018: ["08-24"], 2019: ["08-23"],
    2020: ["08-27"], 2021: ["08-27"], 2022: ["08-26"], 2023: ["08-25"],
    2024: ["08-23"], 2025: ["08-22"], 2026: ["08-28"],
}


def _first_friday(year: int, month: int) -> pd.Timestamp:
    """指定年月の第1金曜日を返す。"""
    first = pd.Timestamp(year=year, month=month, day=1)
    offset = (4 - first.dayofweek) % 7
    return first + pd.Timedelta(days=offset)


def _first_business_day(year: int, month: int) -> pd.Timestamp:
    """指定年月の第1営業日を返す。"""
    first = pd.Timestamp(year=year, month=month, day=1)
    if first.dayofweek >= 5:
        first += pd.offsets.BDay(1)
    return first


def _mid_month(year: int, month: int, day: int = 13) -> pd.Timestamp:
    """指定年月の中旬営業日を返す。"""
    d = pd.Timestamp(year=year, month=month, day=day)
    if d.dayofweek >= 5:
        d += pd.offsets.BDay(1)
    return d


def get_all_event_dates(year: int) -> pd.DataFrame:
    """指定年の全経済イベント日程を返す。

    Returns:
        DataFrame with columns: date (Timestamp), event_type (str)
    """
    events = []

    # FOMC
    for md in FOMC_DATES.get(year, []):
        events.append((pd.Timestamp(f"{year}-{md}"), "FOMC"))

    # 日銀
    for md in BOJ_DATES.get(year, []):
        events.append((pd.Timestamp(f"{year}-{md}"), "BOJ"))

    # ジャクソンホール
    for md in JACKSON_HOLE.get(year, []):
        events.append((pd.Timestamp(f"{year}-{md}"), "JACKSON_HOLE"))

    # NFP: 毎月第1金曜日
    for m in range(1, 13):
        events.append((_first_friday(year, m), "NFP"))

    # CPI: 毎月中旬
    for m in range(1, 13):
        events.append((_mid_month(year, m, 13), "CPI"))

    # GDP: 四半期末月の月末付近
    for m in [1, 4, 7, 10]:
        d = pd.Timestamp(year=year, month=m, day=28)
        if d.dayofweek >= 5:
            d -= pd.offsets.BDay(1)
        events.append((d, "GDP"))

    # ISM: 毎月第1営業日
    for m in range(1, 13):
        events.append((_first_business_day(year, m), "ISM"))

    # 小売売上: 毎月中旬
    for m in range(1, 13):
        events.append((_mid_month(year, m, 15), "RETAIL"))

    df = pd.DataFrame(events, columns=["date", "event_type"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _get_events_for_range(start_year: int, end_year: int) -> pd.DataFrame:
    """複数年のイベントを結合して返す。"""
    frames = [get_all_event_dates(y) for y in range(start_year, end_year + 1)]
    return pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)


def compute_event_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """日付インデックスに対してイベント特徴量を算出する。

    Args:
        index: 営業日の DatetimeIndex

    Returns:
        DataFrame with columns:
            days_to_next_major_event, days_from_last_major_event,
            event_type_next, is_event_day, event_density_past_5d
    """
    start_year = index.min().year - 1
    end_year = index.max().year + 1
    all_events = _get_events_for_range(start_year, end_year)
    event_dates = all_events["date"].values

    bdays = pd.bdate_range(index.min() - pd.Timedelta(days=10), index.max() + pd.Timedelta(days=60))

    results = []
    for d in index:
        d_ts = pd.Timestamp(d)

        # days_to_next_major_event
        future = all_events[all_events["date"] >= d_ts]
        if len(future) > 0:
            next_event_date = future.iloc[0]["date"]
            next_event_type = future.iloc[0]["event_type"]
            days_to = np.busday_count(
                d_ts.date(), pd.Timestamp(next_event_date).date()
            )
        else:
            days_to = 30
            next_event_type = "NONE"

        # days_from_last_major_event
        past = all_events[all_events["date"] <= d_ts]
        if len(past) > 0:
            last_event_date = past.iloc[-1]["date"]
            days_from = np.busday_count(
                pd.Timestamp(last_event_date).date(), d_ts.date()
            )
        else:
            days_from = 30

        # is_event_day
        is_event = int(d_ts.normalize() in set(pd.DatetimeIndex(event_dates).normalize()))

        # event_density_past_5d: t-5d 〜 t-1d のイベント数
        past_5d_start = d_ts - pd.offsets.BDay(5)
        past_5d_events = all_events[
            (all_events["date"] >= past_5d_start) & (all_events["date"] < d_ts)
        ]
        density = len(past_5d_events)

        results.append({
            "days_to_next_major_event": max(days_to, 0),
            "days_from_last_major_event": max(days_from, 0),
            "event_type_next": next_event_type,
            "is_event_day": is_event,
            "event_density_past_5d": density,
        })

    return pd.DataFrame(results, index=index)
```

- [ ] **Step 4: テスト実行、パスを確認**

```bash
uv run pytest tests/test_events.py -v
```

Expected: ALL PASSED

- [ ] **Step 5: コミット**

```bash
git add data/events.py tests/test_events.py
git commit -m "feat: 経済イベントカレンダー（FOMC, NFP, CPI等のイベント特徴量算出）"
```

---

### Task 3: データ取得 (`data/fetch.py`)

**Files:**
- Create: `data/fetch.py`
- Create: `tests/test_fetch.py`

- [ ] **Step 1: tests/test_fetch.py にテストを作成**

```python
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
```

- [ ] **Step 2: テスト実行、失敗を確認**

```bash
uv run pytest tests/test_fetch.py -v
```

Expected: ImportError

- [ ] **Step 3: data/fetch.py を実装**

```python
"""Yahoo Finance / FRED からの生データ取得。

取得したデータは artifacts/raw_data.parquet にキャッシュされ、
当日中は再取得しない。
"""

import os
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
    RAW_DATA_PATH,
    PROJECT_ROOT,
)

load_dotenv(PROJECT_ROOT / ".env")


def fetch_yahoo_data(ticker: str, years: int = DATA_YEARS) -> pd.DataFrame:
    """Yahoo Finance から日足データを取得する。"""
    end = date.today()
    start = end - timedelta(days=years * 365)
    df = yf.download(ticker, start=str(start), end=str(end), progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()


def fetch_fred_data(series_id: str, years: int = DATA_YEARS) -> pd.Series:
    """FRED API から経済指標を取得する。"""
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise ValueError("FRED_API_KEY not found in environment")
    fred = Fred(api_key=api_key)
    end = date.today()
    start = end - timedelta(days=years * 365)
    series = fred.get_series(series_id, observation_start=str(start), observation_end=str(end))
    return series.dropna()


def fetch_all_data(years: int = DATA_YEARS, use_cache: bool = True) -> pd.DataFrame:
    """全データソースから取得し、USD/JPY の取引日ベースでマージする。

    Args:
        years: 取得年数
        use_cache: True なら当日キャッシュを利用

    Returns:
        マージ済み DataFrame（index=取引日）
    """
    # キャッシュチェック
    if use_cache and RAW_DATA_PATH.exists():
        mtime = date.fromtimestamp(RAW_DATA_PATH.stat().st_mtime)
        if mtime == date.today():
            return pd.read_parquet(RAW_DATA_PATH)

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

    # FRED データ
    for name, series_id in FRED_SERIES.items():
        series = fetch_fred_data(series_id, years)
        fred_df = series.to_frame(name=f"fred_{name}")
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
```

- [ ] **Step 4: テスト実行、パスを確認**

```bash
uv run pytest tests/test_fetch.py -v
```

Expected: ALL PASSED

- [ ] **Step 5: コミット**

```bash
git add data/fetch.py tests/test_fetch.py
git commit -m "feat: Yahoo Finance / FRED データ取得モジュール（キャッシュ付き）"
```

---

### Task 4: 特徴量エンジニアリング (`data/features.py`)

**Files:**
- Create: `data/features.py`
- Create: `tests/test_features.py`

- [ ] **Step 1: tests/test_features.py にテストを作成**

```python
"""data/features.py のテスト。"""

import pandas as pd
import numpy as np
import pytest
from data.features import (
    apply_publication_lag,
    compute_technical_features,
    compute_market_returns,
    compute_macro_features,
    compute_calendar_features,
    build_features,
)


class TestApplyPublicationLag:
    def test_shifts_values_by_lag_days(self):
        index = pd.bdate_range("2025-01-02", periods=60)
        series = pd.Series(range(60), index=index)
        lagged = apply_publication_lag(series, lag_days=5)
        # ラグ適用後、先頭5営業日分は NaN になるはず
        assert lagged.iloc[:5].isna().all()
        # ラグ適用後の値はオリジナルのシフト版
        assert lagged.iloc[5] == series.iloc[0]

    def test_zero_lag_returns_original(self):
        index = pd.bdate_range("2025-01-02", periods=10)
        series = pd.Series(range(10), index=index)
        lagged = apply_publication_lag(series, lag_days=0)
        pd.testing.assert_series_equal(lagged, series)


class TestComputeTechnicalFeatures:
    def test_output_columns(self, sample_ohlcv):
        result = compute_technical_features(sample_ohlcv)
        expected = {"sma_5", "sma_20", "sma_60", "rsi_14", "macd", "bb_upper", "bb_lower", "atr"}
        assert expected.issubset(set(result.columns))

    def test_no_nan_after_warmup(self, sample_ohlcv):
        result = compute_technical_features(sample_ohlcv)
        # 60日以降は NaN なし（SMA(60) のウォームアップ）
        assert result.iloc[60:].isna().sum().sum() == 0


class TestComputeMarketReturns:
    def test_output_columns(self, sample_ohlcv):
        df = pd.DataFrame({
            "usdjpy_close": sample_ohlcv["Close"],
            "sp500_close": sample_ohlcv["Close"] * 30,
            "nikkei_close": sample_ohlcv["Close"] * 250,
            "vix_close": 20 + np.random.randn(len(sample_ohlcv)),
            "oil_close": 70 + np.random.randn(len(sample_ohlcv)),
            "gold_close": 2000 + np.random.randn(len(sample_ohlcv)),
        }, index=sample_ohlcv.index)
        result = compute_market_returns(df)
        assert "log_return" in result.columns
        assert "sp500_return" in result.columns


class TestComputeCalendarFeatures:
    def test_output_columns(self):
        index = pd.bdate_range("2025-01-02", periods=30)
        result = compute_calendar_features(index)
        assert "day_of_week" in result.columns
        assert "month" in result.columns
        assert "is_month_end" in result.columns

    def test_day_of_week_range(self):
        index = pd.bdate_range("2025-01-02", periods=30)
        result = compute_calendar_features(index)
        assert result["day_of_week"].between(0, 4).all()


class TestBuildFeatures:
    def test_returns_complete_dataframe(self, sample_ohlcv):
        raw = pd.DataFrame({
            "usdjpy_open": sample_ohlcv["Open"],
            "usdjpy_high": sample_ohlcv["High"],
            "usdjpy_low": sample_ohlcv["Low"],
            "usdjpy_close": sample_ohlcv["Close"],
            "usdjpy_volume": sample_ohlcv["Volume"],
            "sp500_close": sample_ohlcv["Close"] * 30,
            "nikkei_close": sample_ohlcv["Close"] * 250,
            "vix_close": 20 + np.random.randn(len(sample_ohlcv)),
            "oil_close": 70 + np.random.randn(len(sample_ohlcv)),
            "gold_close": 2000 + np.random.randn(len(sample_ohlcv)),
            "fred_us_10y": np.full(len(sample_ohlcv), 4.5),
            "fred_jp_10y": np.full(len(sample_ohlcv), 1.0),
            "fred_ff_rate": np.full(len(sample_ohlcv), 5.25),
            "fred_cpi": np.full(len(sample_ohlcv), 310.0),
            "fred_unemployment": np.full(len(sample_ohlcv), 3.7),
            "fred_gdp": np.full(len(sample_ohlcv), 28000.0),
            "fred_m2": np.full(len(sample_ohlcv), 21000.0),
            "fred_dxy": np.full(len(sample_ohlcv), 104.0),
        }, index=sample_ohlcv.index)
        result = build_features(raw)
        assert "log_return" in result.columns
        assert len(result) > 0
        # NaN が残っていないことを確認
        assert result.isna().sum().sum() == 0
```

- [ ] **Step 2: テスト実行、失敗を確認**

```bash
uv run pytest tests/test_features.py -v
```

Expected: ImportError

- [ ] **Step 3: data/features.py を実装**

```python
"""特徴量エンジニアリング。

テクニカル指標、関連市場リターン、マクロ経済指標（公表ラグ適用済み）、
カレンダー特徴量、経済イベント特徴量を統合する。
"""

import numpy as np
import pandas as pd
import ta

from config import PUBLICATION_LAGS
from data.events import compute_event_features


def apply_publication_lag(series: pd.Series, lag_days: int) -> pd.Series:
    """公表ラグを適用する。

    営業日ベースで lag_days 分シフトし、公表前のデータが
    モデルに漏れることを防ぐ。学習時・推論時で同じ関数を呼ぶ。
    """
    if lag_days <= 0:
        return series
    return series.shift(lag_days, freq="B").reindex(series.index)


def compute_technical_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """OHLCV からテクニカル指標を算出する。

    Args:
        ohlcv: columns = Open, High, Low, Close, Volume (or usdjpy_*)
    """
    # カラム名の正規化
    close = ohlcv.get("Close", ohlcv.get("usdjpy_close"))
    high = ohlcv.get("High", ohlcv.get("usdjpy_high"))
    low = ohlcv.get("Low", ohlcv.get("usdjpy_low"))

    result = pd.DataFrame(index=ohlcv.index)

    # SMA
    result["sma_5"] = ta.trend.sma_indicator(close, window=5)
    result["sma_20"] = ta.trend.sma_indicator(close, window=20)
    result["sma_60"] = ta.trend.sma_indicator(close, window=60)

    # RSI
    result["rsi_14"] = ta.momentum.rsi(close, window=14)

    # MACD
    result["macd"] = ta.trend.macd_diff(close)

    # ボリンジャーバンド
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    result["bb_upper"] = bb.bollinger_hband()
    result["bb_lower"] = bb.bollinger_lband()

    # ATR
    result["atr"] = ta.volatility.average_true_range(high, low, close, window=14)

    return result


def compute_market_returns(df: pd.DataFrame) -> pd.DataFrame:
    """対数リターンを算出する。"""
    result = pd.DataFrame(index=df.index)

    # USD/JPY リターン
    result["log_return"] = np.log(df["usdjpy_close"] / df["usdjpy_close"].shift(1))
    result["log_return_5d"] = np.log(df["usdjpy_close"] / df["usdjpy_close"].shift(5))
    result["log_return_20d"] = np.log(df["usdjpy_close"] / df["usdjpy_close"].shift(20))

    # 関連市場リターン
    for name in ["sp500", "nikkei", "vix", "oil", "gold"]:
        col = f"{name}_close"
        if col in df.columns:
            result[f"{name}_return"] = np.log(df[col] / df[col].shift(1))

    return result


def compute_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """マクロ経済指標に公表ラグを適用して特徴量化する。"""
    result = pd.DataFrame(index=df.index)

    # 各 FRED 指標にラグを適用
    for name, lag in PUBLICATION_LAGS.items():
        col = f"fred_{name}"
        if col in df.columns:
            result[col] = apply_publication_lag(df[col], lag)

    # 日米金利差
    if "fred_us_10y" in result.columns and "fred_jp_10y" in result.columns:
        result["rate_diff"] = result["fred_us_10y"] - result["fred_jp_10y"]

    return result


def compute_calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """カレンダー特徴量を算出する。"""
    result = pd.DataFrame(index=index)
    result["day_of_week"] = index.dayofweek
    result["month"] = index.month
    result["is_month_end"] = index.is_month_end.astype(int)
    return result


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """生データから全特徴量を構築する。

    Args:
        raw: fetch_all_data() の出力

    Returns:
        全特徴量を含む DataFrame（NaN 行は除去済み）
    """
    ohlcv = raw[["usdjpy_open", "usdjpy_high", "usdjpy_low", "usdjpy_close", "usdjpy_volume"]]
    ohlcv_renamed = ohlcv.rename(columns={
        "usdjpy_open": "Open", "usdjpy_high": "High",
        "usdjpy_low": "Low", "usdjpy_close": "Close",
        "usdjpy_volume": "Volume",
    })

    technical = compute_technical_features(ohlcv_renamed)
    returns = compute_market_returns(raw)
    macro = compute_macro_features(raw)
    calendar = compute_calendar_features(raw.index)
    events = compute_event_features(raw.index)

    # 全結合
    features = pd.concat([raw, technical, returns, macro, calendar, events], axis=1)

    # 重複カラムを除去
    features = features.loc[:, ~features.columns.duplicated()]

    # NaN を除去（SMA(60) のウォームアップ等）
    features = features.dropna()

    return features
```

- [ ] **Step 4: テスト実行、パスを確認**

```bash
uv run pytest tests/test_features.py -v
```

Expected: ALL PASSED

- [ ] **Step 5: コミット**

```bash
git add data/features.py tests/test_features.py
git commit -m "feat: 特徴量エンジニアリング（テクニカル, マクロ公表ラグ, カレンダー, イベント統合）"
```

---

### Task 5: TimeSeriesDataSet 構築 (`data/dataset.py`)

**Files:**
- Create: `data/dataset.py`
- Create: `tests/test_dataset.py`

- [ ] **Step 1: tests/test_dataset.py にテストを作成**

```python
"""data/dataset.py のテスト。"""

import pandas as pd
import numpy as np
import pytest
from data.dataset import prepare_data, create_datasets, split_data


class TestPrepareData:
    def test_adds_time_idx_and_group(self):
        n = 200
        index = pd.bdate_range("2024-01-02", periods=n)
        df = pd.DataFrame({
            "log_return": np.random.randn(n) * 0.01,
            "sma_5": np.random.randn(n),
            "day_of_week": index.dayofweek,
            "month": index.month,
            "is_month_end": index.is_month_end.astype(int),
            "days_to_next_major_event": np.random.randint(0, 20, n),
            "event_type_next": np.random.choice(["FOMC", "NFP", "CPI"], n),
        }, index=index)
        result = prepare_data(df)
        assert "time_idx" in result.columns
        assert "group_id" in result.columns
        assert result["time_idx"].is_monotonic_increasing

    def test_no_nan_in_output(self):
        n = 200
        index = pd.bdate_range("2024-01-02", periods=n)
        df = pd.DataFrame({
            "log_return": np.random.randn(n) * 0.01,
            "sma_5": np.random.randn(n),
            "day_of_week": index.dayofweek,
            "month": index.month,
            "is_month_end": index.is_month_end.astype(int),
            "days_to_next_major_event": np.random.randint(0, 20, n),
            "event_type_next": np.random.choice(["FOMC", "NFP", "CPI"], n),
        }, index=index)
        result = prepare_data(df)
        # group_id と event_type_next 以外に NaN がないこと
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        assert result[numeric_cols].isna().sum().sum() == 0


class TestSplitData:
    def test_split_sizes(self):
        n = 1000
        index = pd.bdate_range("2020-01-02", periods=n)
        df = pd.DataFrame({"time_idx": range(n), "group_id": "usdjpy"}, index=index)
        train, val, tune, test = split_data(df)
        assert len(train) + len(val) + len(tune) + len(test) == n
        assert len(train) > len(val)
        assert len(test) > len(tune)

    def test_temporal_order_preserved(self):
        n = 1000
        index = pd.bdate_range("2020-01-02", periods=n)
        df = pd.DataFrame({"time_idx": range(n), "group_id": "usdjpy"}, index=index)
        train, val, tune, test = split_data(df)
        assert train.index.max() < val.index.min()
        assert val.index.max() < tune.index.min()
        assert tune.index.max() < test.index.min()
```

- [ ] **Step 2: テスト実行、失敗を確認**

```bash
uv run pytest tests/test_dataset.py -v
```

Expected: ImportError

- [ ] **Step 3: data/dataset.py を実装**

```python
"""pytorch-forecasting 用 TimeSeriesDataSet 構築。

build_features() の出力を受け取り、TFT 学習用のデータセットを構築する。
"""

import pandas as pd
import numpy as np
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

from config import (
    ENCODER_LENGTH,
    PREDICTION_LENGTH,
    DATA_SPLIT_RATIOS,
)


# TFT の特徴量分類
TIME_VARYING_KNOWN_CATEGORICALS = ["event_type_next"]
TIME_VARYING_KNOWN_REALS = [
    "day_of_week", "month", "is_month_end",
    "days_to_next_major_event", "days_from_last_major_event",
    "is_event_day", "event_density_past_5d",
]
TIME_VARYING_UNKNOWN_REALS = [
    "log_return",
    "log_return_5d", "log_return_20d",
    "sma_5", "sma_20", "sma_60",
    "rsi_14", "macd", "bb_upper", "bb_lower", "atr",
    "sp500_return", "nikkei_return", "vix_return", "oil_return", "gold_return",
    "fred_us_10y", "fred_jp_10y", "fred_ff_rate",
    "fred_cpi", "fred_unemployment", "fred_gdp", "fred_m2", "fred_dxy",
    "rate_diff",
]


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """TimeSeriesDataSet 用にデータを整形する。

    time_idx（連番）と group_id（単一通貨ペア）を追加する。
    """
    result = df.copy()
    result["time_idx"] = range(len(result))
    result["group_id"] = "usdjpy"

    # カテゴリカルを文字列に変換
    for col in TIME_VARYING_KNOWN_CATEGORICALS:
        if col in result.columns:
            result[col] = result[col].astype(str)

    return result


def split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """時系列順に train/val/threshold_tune/test に分割する。"""
    n = len(df)
    ratios = DATA_SPLIT_RATIOS
    train_end = int(n * ratios["train"])
    val_end = train_end + int(n * ratios["val"])
    tune_end = val_end + int(n * ratios["threshold_tune"])

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    tune = df.iloc[val_end:tune_end]
    test = df.iloc[tune_end:]

    return train, val, tune, test


def create_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
    """学習用・バリデーション用の TimeSeriesDataSet を作成する。

    Returns:
        (training_dataset, validation_dataset)
    """
    # 利用可能な特徴量のみフィルタ
    available_known_reals = [c for c in TIME_VARYING_KNOWN_REALS if c in train_df.columns]
    available_unknown_reals = [c for c in TIME_VARYING_UNKNOWN_REALS if c in train_df.columns]
    available_known_cats = [c for c in TIME_VARYING_KNOWN_CATEGORICALS if c in train_df.columns]

    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="log_return",
        group_ids=["group_id"],
        max_encoder_length=ENCODER_LENGTH,
        max_prediction_length=PREDICTION_LENGTH,
        time_varying_known_reals=available_known_reals,
        time_varying_unknown_reals=available_unknown_reals,
        time_varying_known_categoricals=available_known_cats,
        target_normalizer=GroupNormalizer(groups=["group_id"]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    validation = TimeSeriesDataSet.from_dataset(training, val_df, stop_randomization=True)

    return training, validation
```

- [ ] **Step 4: テスト実行、パスを確認**

```bash
uv run pytest tests/test_dataset.py -v
```

Expected: ALL PASSED

- [ ] **Step 5: コミット**

```bash
git add data/dataset.py tests/test_dataset.py
git commit -m "feat: TimeSeriesDataSet 構築（特徴量分類, データ分割, TFT用整形）"
```

---

### Task 6: カスタム損失関数 (`model/loss.py`)

**Files:**
- Create: `model/loss.py`
- Create: `tests/test_loss.py`

- [ ] **Step 1: tests/test_loss.py にテストを作成**

```python
"""model/loss.py のテスト。"""

import torch
import pytest
from model.loss import DirectionAwareQuantileLoss


class TestDirectionAwareQuantileLoss:
    def setup_method(self):
        self.loss_fn = DirectionAwareQuantileLoss(
            quantiles=[0.1, 0.25, 0.5, 0.75, 0.9],
            direction_weight=1.0,
            smoothing_temperature=0.1,
            dead_zone=1e-4,
        )

    def test_output_shape(self):
        # y_pred: (batch, horizon, quantiles)
        y_pred = torch.randn(32, 5, 5)
        target = torch.randn(32, 5)
        loss = self.loss_fn.loss(y_pred, target)
        assert loss.shape == y_pred.shape

    def test_correct_direction_lower_loss(self):
        """方向が正しい予測は、間違った予測より損失が小さい。"""
        target = torch.tensor([[0.01, 0.02, 0.01, 0.01, 0.01]])  # 上昇

        # 正しい方向の予測
        correct = torch.zeros(1, 5, 5)
        correct[:, :, 2] = 0.01  # median = positive

        # 間違った方向の予測
        wrong = torch.zeros(1, 5, 5)
        wrong[:, :, 2] = -0.01  # median = negative

        loss_correct = self.loss_fn.loss(correct, target).mean()
        loss_wrong = self.loss_fn.loss(wrong, target).mean()
        assert loss_correct < loss_wrong

    def test_dead_zone_suppresses_penalty(self):
        """ターゲットがデッドゾーン内なら方向ペナルティは抑制される。"""
        tiny_target = torch.tensor([[1e-5, 1e-5, 1e-5, 1e-5, 1e-5]])
        y_pred = torch.zeros(1, 5, 5)
        y_pred[:, :, 2] = -1e-5  # 逆方向だがターゲットが微小

        loss_fn_no_dir = DirectionAwareQuantileLoss(
            quantiles=[0.1, 0.25, 0.5, 0.75, 0.9],
            direction_weight=0.0,  # 方向ペナルティなし
        )
        loss_with = self.loss_fn.loss(y_pred, tiny_target).mean()
        loss_without = loss_fn_no_dir.loss(y_pred, tiny_target).mean()
        # デッドゾーン内なのでほぼ同じ
        assert abs(loss_with.item() - loss_without.item()) < 0.01

    def test_gradient_flows(self):
        """勾配が正常に流れることを確認。"""
        y_pred = torch.randn(8, 5, 5, requires_grad=True)
        target = torch.randn(8, 5)
        loss = self.loss_fn.loss(y_pred, target).mean()
        loss.backward()
        assert y_pred.grad is not None
        assert not torch.isnan(y_pred.grad).any()
```

- [ ] **Step 2: テスト実行、失敗を確認**

```bash
uv run pytest tests/test_loss.py -v
```

Expected: ImportError

- [ ] **Step 3: model/loss.py を実装**

```python
"""カスタム損失関数。

DirectionAwareQuantileLoss: QuantileLoss にスムーズな方向ペナルティを追加。
tanh スムージングでゼロ付近の勾配不連続を回避する。
"""

import torch
from pytorch_forecasting.metrics import QuantileLoss

from config import DIRECTION_WEIGHT, SMOOTHING_TEMPERATURE, DEAD_ZONE, QUANTILES


class DirectionAwareQuantileLoss(QuantileLoss):
    """QuantileLoss + 方向ペナルティ。

    中央値予測と実績の方向が不一致の場合にペナルティを課す。
    tanh でスムージングし、デッドゾーンで微小リターンのペナルティを抑制。
    """

    def __init__(
        self,
        quantiles: list[float] | None = None,
        direction_weight: float = DIRECTION_WEIGHT,
        smoothing_temperature: float = SMOOTHING_TEMPERATURE,
        dead_zone: float = DEAD_ZONE,
        **kwargs,
    ):
        super().__init__(quantiles=quantiles or QUANTILES, **kwargs)
        self.direction_weight = direction_weight
        self.smoothing_temperature = smoothing_temperature
        self.dead_zone = dead_zone

    def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """損失を計算する。

        Args:
            y_pred: (batch, horizon, n_quantiles)
            target: (batch, horizon)

        Returns:
            (batch, horizon, n_quantiles) の損失テンソル
        """
        # 標準 quantile loss
        ql = super().loss(y_pred, target)

        if self.direction_weight <= 0:
            return ql

        # 中央値予測 (q50) を取得
        q_mid = y_pred.size(-1) // 2
        pred_median = y_pred[..., q_mid]

        # tanh スムージング: 連続的な方向スコア [-1, 1]
        pred_dir = torch.tanh(pred_median / self.smoothing_temperature)
        target_dir = torch.tanh(target / self.smoothing_temperature)

        # 方向不一致ペナルティ: (1 - pred_dir * target_dir) / 2 → [0, 1]
        mismatch = (1 - pred_dir * target_dir) / 2

        # デッドゾーン: ターゲットが微小な場合はペナルティを抑制
        dead_zone_mask = (target.abs() < self.dead_zone).float()
        mismatch = mismatch * (1 - dead_zone_mask)

        # QL と同じ形状に拡張して加算
        dir_penalty = mismatch.unsqueeze(-1) * self.direction_weight

        return ql + dir_penalty
```

- [ ] **Step 4: テスト実行、パスを確認**

```bash
uv run pytest tests/test_loss.py -v
```

Expected: ALL PASSED

- [ ] **Step 5: コミット**

```bash
git add model/loss.py tests/test_loss.py
git commit -m "feat: DirectionAwareQuantileLoss（tanhスムージング+デッドゾーン付き方向ペナルティ）"
```

---

### Task 7: 学習パイプライン (`model/trainer.py`)

**Files:**
- Create: `model/trainer.py`
- Create: `tests/test_trainer.py`

- [ ] **Step 1: tests/test_trainer.py にテストを作成**

```python
"""model/trainer.py のテスト。"""

import pytest
from model.trainer import build_trainer, build_tft


class TestBuildTrainer:
    def test_returns_trainer(self):
        trainer = build_trainer(max_epochs=1, fast_dev_run=True)
        assert trainer is not None
        assert trainer.max_epochs == 1


class TestBuildTft:
    def test_returns_tft_with_correct_params(self):
        import pandas as pd
        import numpy as np
        from data.dataset import prepare_data, create_datasets

        n = 200
        index = pd.bdate_range("2024-01-02", periods=n)
        df = pd.DataFrame({
            "log_return": np.random.randn(n) * 0.01,
            "sma_5": np.random.randn(n),
            "day_of_week": index.dayofweek,
            "month": index.month,
            "is_month_end": index.is_month_end.astype(int),
            "days_to_next_major_event": np.random.randint(0, 20, n),
            "days_from_last_major_event": np.random.randint(0, 20, n),
            "event_type_next": np.random.choice(["FOMC", "NFP", "CPI"], n),
            "is_event_day": np.random.randint(0, 2, n),
            "event_density_past_5d": np.random.randint(0, 3, n),
        }, index=index)
        prepped = prepare_data(df)
        training, _ = create_datasets(prepped.iloc[:160], prepped.iloc[160:])
        tft = build_tft(training)
        assert tft is not None
```

- [ ] **Step 2: テスト実行、失敗を確認**

```bash
uv run pytest tests/test_trainer.py -v
```

Expected: ImportError

- [ ] **Step 3: model/trainer.py を実装**

```python
"""Lightning Trainer 設定と TFT モデル構築。"""

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

from config import (
    ARTIFACT_DIR,
    MAX_EPOCHS,
    EARLY_STOP_PATIENCE,
    LEARNING_RATE,
    HIDDEN_SIZE,
    ATTENTION_HEAD_SIZE,
    DROPOUT,
    HIDDEN_CONTINUOUS_SIZE,
    OUTPUT_SIZE,
    TOP_K_CHECKPOINTS,
)
from model.loss import DirectionAwareQuantileLoss

# GPU 最適化
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("medium")
    torch.backends.cudnn.benchmark = True


def build_trainer(
    max_epochs: int = MAX_EPOCHS,
    fast_dev_run: bool = False,
) -> pl.Trainer:
    """Lightning Trainer を構築する。"""
    ckpt_dir = ARTIFACT_DIR / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=EARLY_STOP_PATIENCE, mode="min"),
        LearningRateMonitor(logging_interval="epoch"),
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="tft-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=TOP_K_CHECKPOINTS,
        ),
    ]

    return pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        callbacks=callbacks,
        gradient_clip_val=0.1,
        fast_dev_run=fast_dev_run,
        enable_progress_bar=True,
    )


def build_tft(training_dataset: TimeSeriesDataSet) -> TemporalFusionTransformer:
    """TFT モデルを構築する。"""
    return TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=LEARNING_RATE,
        hidden_size=HIDDEN_SIZE,
        attention_head_size=ATTENTION_HEAD_SIZE,
        dropout=DROPOUT,
        hidden_continuous_size=HIDDEN_CONTINUOUS_SIZE,
        output_size=OUTPUT_SIZE,
        loss=DirectionAwareQuantileLoss(),
        reduce_on_plateau_patience=4,
    )
```

- [ ] **Step 4: テスト実行、パスを確認**

```bash
uv run pytest tests/test_trainer.py -v
```

Expected: ALL PASSED

- [ ] **Step 5: コミット**

```bash
git add model/trainer.py tests/test_trainer.py
git commit -m "feat: Lightning Trainer + TFT モデル構築パイプライン"
```

---

### Task 8: 学習エントリポイント (`scripts/train.py`)

**Files:**
- Create: `scripts/train.py`

- [ ] **Step 1: scripts/train.py を実装**

```python
"""学習実行エントリポイント。

Usage:
    uv run python scripts/train.py                # 通常学習
    uv run python scripts/train.py --optuna       # Optunaハイパラチューニング
"""

import argparse
import json
import warnings

import optuna
import torch
from pytorch_forecasting import TemporalFusionTransformer

from config import (
    ARTIFACT_DIR,
    BATCH_SIZE,
    OPTUNA_DB,
    ENCODER_LENGTH,
    PREDICTION_LENGTH,
)
from data.fetch import fetch_all_data
from data.features import build_features
from data.dataset import prepare_data, split_data, create_datasets
from model.trainer import build_trainer, build_tft
from model.loss import DirectionAwareQuantileLoss

warnings.filterwarnings("ignore", ".*does not have many workers.*")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = DEVICE.type == "cuda"


def train_once() -> None:
    """通常の学習を1回実行する。"""
    print("=== データ取得 ===")
    raw = fetch_all_data()

    print("=== 特徴量構築 ===")
    features = build_features(raw)

    print("=== データセット構築 ===")
    prepped = prepare_data(features)
    train_df, val_df, tune_df, test_df = split_data(prepped)

    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, "
          f"Tune: {len(tune_df)}, Test: {len(test_df)}")

    training, validation = create_datasets(train_df, val_df)

    train_loader = training.to_dataloader(
        train=True, batch_size=BATCH_SIZE, num_workers=0, pin_memory=PIN_MEMORY,
    )
    val_loader = validation.to_dataloader(
        train=False, batch_size=BATCH_SIZE, num_workers=0, pin_memory=PIN_MEMORY,
    )

    print("=== モデル構築 ===")
    tft = build_tft(training)
    print(f"  Parameters: {tft.size() / 1e3:.1f}k")

    print("=== 学習開始 ===")
    trainer = build_trainer()
    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print(f"=== 学習完了 ===")
    print(f"  Best model: {trainer.checkpoint_callback.best_model_path}")

    # 学習メタデータ保存
    meta = {
        "train_size": len(train_df),
        "val_size": len(val_df),
        "tune_size": len(tune_df),
        "test_size": len(test_df),
        "best_val_loss": float(trainer.checkpoint_callback.best_model_score),
        "best_model_path": trainer.checkpoint_callback.best_model_path,
        "encoder_length": ENCODER_LENGTH,
        "prediction_length": PREDICTION_LENGTH,
    }
    meta_path = ARTIFACT_DIR / "train_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"  Metadata saved to {meta_path}")


def train_optuna(n_trials: int = 50) -> None:
    """Optuna でハイパーパラメータチューニングを行う。"""
    print("=== データ取得 ===")
    raw = fetch_all_data()
    features = build_features(raw)
    prepped = prepare_data(features)
    train_df, val_df, _, _ = split_data(prepped)

    def objective(trial: optuna.Trial) -> float:
        hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128])
        dropout = trial.suggest_float("dropout", 0.1, 0.4, step=0.05)
        lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        direction_weight = trial.suggest_float("direction_weight", 0.0, 3.0, step=0.5)

        training, validation = create_datasets(train_df, val_df)
        train_loader = training.to_dataloader(
            train=True, batch_size=BATCH_SIZE, num_workers=0,
        )
        val_loader = validation.to_dataloader(
            train=False, batch_size=BATCH_SIZE, num_workers=0,
        )

        tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=lr,
            hidden_size=hidden_size,
            dropout=dropout,
            hidden_continuous_size=hidden_size // 2,
            output_size=5,
            loss=DirectionAwareQuantileLoss(direction_weight=direction_weight),
            reduce_on_plateau_patience=4,
        )

        trainer = build_trainer(max_epochs=30)
        trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

        return float(trainer.checkpoint_callback.best_model_score)

    study = optuna.create_study(
        direction="minimize",
        storage=f"sqlite:///{OPTUNA_DB}",
        study_name="tft_usdjpy",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials)

    print(f"=== Optuna 完了 ===")
    print(f"  Best trial: {study.best_trial.number}")
    print(f"  Best val_loss: {study.best_value:.6f}")
    print(f"  Best params: {study.best_params}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TFT USD/JPY 学習")
    parser.add_argument("--optuna", action="store_true", help="Optunaチューニング")
    parser.add_argument("--n-trials", type=int, default=50, help="Optuna試行回数")
    args = parser.parse_args()

    if args.optuna:
        train_optuna(args.n_trials)
    else:
        train_once()
```

- [ ] **Step 2: スモークテスト（dry-run）**

```bash
uv run python scripts/train.py --help
```

Expected: ヘルプメッセージが表示される

- [ ] **Step 3: コミット**

```bash
git add scripts/train.py
git commit -m "feat: 学習エントリポイント（通常学習 + Optuna ハイパラチューニング）"
```

---

### Task 9: 評価スクリプト (`scripts/evaluate.py`)

**Files:**
- Create: `scripts/evaluate.py`

- [ ] **Step 1: scripts/evaluate.py を実装**

```python
"""モデル評価・可視化スクリプト。

保存済みチェックポイントからモデルをロードし、テストデータで評価する。
動的閾値チューニングも行う。

Usage:
    uv run python scripts/evaluate.py
    uv run python scripts/evaluate.py --top-k 5
"""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from pytorch_forecasting import TemporalFusionTransformer

from config import (
    ARTIFACT_DIR,
    BATCH_SIZE,
    ENCODER_LENGTH,
    PREDICTION_LENGTH,
    TOP_K_CHECKPOINTS,
    QUANTILES,
)
from data.fetch import fetch_all_data
from data.features import build_features
from data.dataset import prepare_data, split_data, create_datasets

warnings.filterwarnings("ignore", ".*does not have many workers.*")

plt.rcParams["font.family"] = "MS Gothic"
plt.rcParams["axes.unicode_minus"] = False

EVAL_DIR = ARTIFACT_DIR / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)


def find_best_checkpoints(top_k: int = TOP_K_CHECKPOINTS) -> list[Path]:
    """val_loss が最小の top-k チェックポイントを返す。"""
    ckpt_dir = ARTIFACT_DIR / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {ckpt_dir}")

    def parse_val_loss(p: Path) -> float:
        name = p.stem
        for part in name.split("-"):
            if part.startswith("val_loss="):
                return float(part.split("=")[1])
        return float("inf")

    ckpts.sort(key=parse_val_loss)
    return ckpts[:top_k]


def ensemble_predict(
    models: list[TemporalFusionTransformer],
    dataloader,
) -> dict:
    """top-k モデルのアンサンブル予測。

    - q50: 平均
    - q10: 各モデルの min
    - q90: 各モデルの max
    """
    all_preds = []
    for model in models:
        preds = model.predict(dataloader, mode="quantiles", return_x=False)
        all_preds.append(preds)

    stacked = torch.stack(all_preds)  # (n_models, batch, horizon, quantiles)
    q_mid = len(QUANTILES) // 2

    result = {
        "median": stacked[:, :, :, q_mid].mean(dim=0),  # q50 平均
        "q10": stacked[:, :, :, 0].min(dim=0).values,    # q10 min
        "q90": stacked[:, :, :, -1].max(dim=0).values,   # q90 max
    }
    return result


def compute_direction_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict:
    """方向精度と方向比率メトリクスを算出する。"""
    actual_up = (actual > 0).astype(float)
    pred_up = (predicted > 0).astype(float)

    accuracy = (actual_up == pred_up).mean()
    actual_up_ratio = actual_up.mean()
    pred_up_ratio = pred_up.mean()
    ratio_gap = abs(actual_up_ratio - pred_up_ratio)

    return {
        "direction_accuracy": float(accuracy),
        "actual_up_ratio": float(actual_up_ratio),
        "pred_up_ratio": float(pred_up_ratio),
        "direction_ratio_gap": float(ratio_gap),
    }


def find_optimal_threshold(
    predictions: np.ndarray,
    actuals: np.ndarray,
) -> float:
    """up:down 比率が実績に最も近くなる閾値を探索する。"""
    actual_up_ratio = (actuals > 0).mean()
    best_threshold = 0.0
    best_gap = float("inf")

    for t in np.linspace(predictions.min(), predictions.max(), 1000):
        pred_up_ratio = (predictions > t).mean()
        gap = abs(actual_up_ratio - pred_up_ratio)
        if gap < best_gap:
            best_gap = gap
            best_threshold = t

    return best_threshold


def evaluate(top_k: int = TOP_K_CHECKPOINTS) -> None:
    """テストデータでモデルを評価する。"""
    print("=== データ準備 ===")
    raw = fetch_all_data()
    features = build_features(raw)
    prepped = prepare_data(features)
    train_df, val_df, tune_df, test_df = split_data(prepped)

    training, _ = create_datasets(train_df, val_df)

    print("=== モデルロード ===")
    ckpts = find_best_checkpoints(top_k)
    print(f"  Using {len(ckpts)} checkpoints")
    models = [TemporalFusionTransformer.load_from_checkpoint(str(p)) for p in ckpts]

    # --- 閾値チューニング (tune セットで実施) ---
    print("=== 閾値チューニング ===")
    tune_dataset = training.from_dataset(training, tune_df, stop_randomization=True)
    tune_loader = tune_dataset.to_dataloader(
        train=False, batch_size=BATCH_SIZE, num_workers=0,
    )
    tune_preds = ensemble_predict(models, tune_loader)

    thresholds = {}
    for h in range(PREDICTION_LENGTH):
        preds_h = tune_preds["median"][:, h].numpy()
        actuals_h = torch.stack([
            y[0] for _, (y, _) in zip(range(len(tune_loader.dataset)), tune_loader.dataset)
        ])[:len(preds_h), h].numpy()
        thresholds[f"horizon_{h+1}"] = find_optimal_threshold(preds_h, actuals_h)

    print(f"  Thresholds: {thresholds}")

    # 閾値を保存
    threshold_path = ARTIFACT_DIR / "thresholds.json"
    with open(threshold_path, "w") as f:
        json.dump(thresholds, f, indent=2)

    # --- テストセットで評価 ---
    print("=== テストセット評価 ===")
    test_dataset = training.from_dataset(training, test_df, stop_randomization=True)
    test_loader = test_dataset.to_dataloader(
        train=False, batch_size=BATCH_SIZE, num_workers=0,
    )
    test_preds = ensemble_predict(models, test_loader)

    report = {"thresholds": thresholds, "horizons": {}}

    for h in range(PREDICTION_LENGTH):
        preds_h = test_preds["median"][:, h].numpy()
        actual_h_list = []
        for _, (y, _) in zip(range(len(test_loader.dataset)), test_loader.dataset):
            actual_h_list.append(y[0])
        actuals_h = torch.stack(actual_h_list)[:len(preds_h), h].numpy()

        mae = float(np.abs(preds_h - actuals_h).mean())
        rmse = float(np.sqrt(((preds_h - actuals_h) ** 2).mean()))
        threshold = thresholds[f"horizon_{h+1}"]
        preds_thresholded = (preds_h > threshold).astype(float)
        dir_metrics = compute_direction_metrics(actuals_h, preds_thresholded)

        report["horizons"][f"horizon_{h+1}"] = {
            "mae": mae,
            "rmse": rmse,
            **dir_metrics,
        }
        print(f"  Horizon {h+1}: MAE={mae:.6f}, RMSE={rmse:.6f}, "
              f"DirAcc={dir_metrics['direction_accuracy']:.3f}, "
              f"RatioGap={dir_metrics['direction_ratio_gap']:.3f}")

    # --- レポート保存 ---
    report_path = EVAL_DIR / "eval_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved to {report_path}")

    # --- 可視化 ---
    # 方向比率比較
    fig, ax = plt.subplots(figsize=(10, 5))
    horizons = list(report["horizons"].keys())
    actual_ratios = [report["horizons"][h]["actual_up_ratio"] for h in horizons]
    pred_ratios = [report["horizons"][h]["pred_up_ratio"] for h in horizons]
    x = range(len(horizons))
    ax.bar([i - 0.15 for i in x], actual_ratios, width=0.3, label="実績", alpha=0.8)
    ax.bar([i + 0.15 for i in x], pred_ratios, width=0.3, label="予測", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i+1}d" for i in x])
    ax.set_ylabel("上昇比率")
    ax.set_title("方向比率: 実績 vs 予測")
    ax.legend()
    fig.savefig(EVAL_DIR / "direction_ratio.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("  Plots saved to", EVAL_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TFT USD/JPY 評価")
    parser.add_argument("--top-k", type=int, default=TOP_K_CHECKPOINTS)
    args = parser.parse_args()
    evaluate(args.top_k)
```

- [ ] **Step 2: スモークテスト**

```bash
uv run python scripts/evaluate.py --help
```

Expected: ヘルプメッセージが表示される

- [ ] **Step 3: コミット**

```bash
git add scripts/evaluate.py
git commit -m "feat: 評価スクリプト（方向比率キャリブレーション + 動的閾値チューニング）"
```

---

### Task 10: 日次予測 (`scripts/predict.py`)

**Files:**
- Create: `scripts/predict.py`

- [ ] **Step 1: scripts/predict.py を実装**

```python
"""日次予測スクリプト。

最新データを取得し、保存済みモデルで5日先まで予測する。
結果は SQLite に蓄積される。

Usage:
    uv run python scripts/predict.py
"""

import json
import sqlite3
import warnings
from datetime import date

import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer

from config import (
    ARTIFACT_DIR,
    BATCH_SIZE,
    ENCODER_LENGTH,
    PREDICTION_LENGTH,
    PREDICTIONS_DB,
    QUANTILES,
)
from data.fetch import fetch_all_data
from data.features import build_features
from data.events import compute_event_features
from data.dataset import (
    prepare_data,
    create_datasets,
    split_data,
    TIME_VARYING_KNOWN_REALS,
    TIME_VARYING_KNOWN_CATEGORICALS,
)
from scripts.evaluate import find_best_checkpoints, ensemble_predict

warnings.filterwarnings("ignore", ".*does not have many workers.*")


def build_decoder_data(last_date: pd.Timestamp, features_df: pd.DataFrame) -> pd.DataFrame:
    """未来5営業日分の decoder_data (time_varying_known) を生成する。

    Args:
        last_date: encoder の最終日
        features_df: 最新の全特徴量 DataFrame

    Returns:
        未来5営業日分の known 特徴量 DataFrame
    """
    future_dates = pd.bdate_range(
        start=last_date + pd.offsets.BDay(1),
        periods=PREDICTION_LENGTH,
    )

    decoder = pd.DataFrame(index=future_dates)

    # カレンダー特徴量
    decoder["day_of_week"] = future_dates.dayofweek
    decoder["month"] = future_dates.month
    decoder["is_month_end"] = future_dates.is_month_end.astype(int)

    # イベント特徴量 (各未来ステップ t を基準日として算出)
    event_features = compute_event_features(future_dates)
    for col in event_features.columns:
        decoder[col] = event_features[col].values

    return decoder


def predict_daily() -> None:
    """日次予測を実行する。"""
    print("=== 最新データ取得 ===")
    raw = fetch_all_data(use_cache=False)
    features = build_features(raw)
    prepped = prepare_data(features)

    # 学習用データセットの構築（normalizer 再利用のため）
    train_df, val_df, _, _ = split_data(prepped)
    training, _ = create_datasets(train_df, val_df)

    # encoder_data: 最新60営業日
    encoder_data = prepped.iloc[-ENCODER_LENGTH:]
    last_date = encoder_data.index[-1]

    # decoder_data: 未来5営業日
    decoder_known = build_decoder_data(last_date, features)

    # encoder + decoder を結合
    # decoder 側には unknown/observed 列を NaN で埋める
    decoder_full = decoder_known.copy()
    for col in prepped.columns:
        if col not in decoder_full.columns:
            if col == "group_id":
                decoder_full[col] = "usdjpy"
            elif col == "time_idx":
                last_idx = encoder_data["time_idx"].iloc[-1]
                decoder_full[col] = range(last_idx + 1, last_idx + 1 + PREDICTION_LENGTH)
            elif col == "log_return":
                decoder_full[col] = 0.0  # ターゲットのプレースホルダー
            else:
                decoder_full[col] = 0.0  # unknown は decoder で不使用だが列が必要

    combined = pd.concat([encoder_data, decoder_full])

    # predict
    print("=== モデルロード ===")
    ckpts = find_best_checkpoints()
    models = [TemporalFusionTransformer.load_from_checkpoint(str(p)) for p in ckpts]

    pred_dataset = training.from_dataset(training, combined, stop_randomization=True)
    pred_loader = pred_dataset.to_dataloader(
        train=False, batch_size=1, num_workers=0,
    )

    preds = ensemble_predict(models, pred_loader)

    # 閾値ロード
    threshold_path = ARTIFACT_DIR / "thresholds.json"
    if threshold_path.exists():
        with open(threshold_path) as f:
            thresholds = json.load(f)
    else:
        thresholds = {f"horizon_{h+1}": 0.0 for h in range(PREDICTION_LENGTH)}

    # 結果を整形
    future_dates = pd.bdate_range(
        start=last_date + pd.offsets.BDay(1),
        periods=PREDICTION_LENGTH,
    )

    results = []
    for h in range(PREDICTION_LENGTH):
        threshold = thresholds.get(f"horizon_{h+1}", 0.0)
        median_val = float(preds["median"][0, h])
        direction = "UP" if median_val > threshold else "DOWN"

        results.append({
            "prediction_date": str(date.today()),
            "target_date": str(future_dates[h].date()),
            "horizon": h + 1,
            "median": median_val,
            "q10": float(preds["q10"][0, h]),
            "q90": float(preds["q90"][0, h]),
            "threshold": threshold,
            "direction": direction,
        })

    results_df = pd.DataFrame(results)
    print("\n=== 予測結果 ===")
    print(results_df.to_string(index=False))

    # SQLite に保存
    PREDICTIONS_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(PREDICTIONS_DB))
    results_df.to_sql("predictions", conn, if_exists="append", index=False)
    conn.close()
    print(f"\n  Saved to {PREDICTIONS_DB}")


if __name__ == "__main__":
    predict_daily()
```

- [ ] **Step 2: スモークテスト**

```bash
uv run python scripts/predict.py --help
```

Expected: エラーなく実行される（help はないがパースはされる）

- [ ] **Step 3: コミット**

```bash
git add scripts/predict.py
git commit -m "feat: 日次予測スクリプト（decoder_data 生成 + SQLite 蓄積 + 閾値適用）"
```

---

### Task 11: Streamlit ダッシュボード (`dashboard/app.py`)

**Files:**
- Create: `dashboard/app.py`

- [ ] **Step 1: dashboard/app.py を実装**

```python
"""Streamlit ダッシュボード。

USD/JPY TFT 予測の結果を可視化する。

Usage:
    uv run streamlit run dashboard/app.py
"""

import json
import sqlite3
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from pytorch_forecasting import TemporalFusionTransformer

from config import ARTIFACT_DIR, PREDICTIONS_DB, PREDICTION_LENGTH
from data.events import get_all_event_dates

st.set_page_config(page_title="USD/JPY TFT 予測", layout="wide")
st.title("USD/JPY TFT 予測ダッシュボード")


@st.cache_data(ttl=3600)
def load_predictions() -> pd.DataFrame:
    """SQLite から予測履歴を読み込む。"""
    if not PREDICTIONS_DB.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(str(PREDICTIONS_DB))
    df = pd.read_sql("SELECT * FROM predictions ORDER BY prediction_date DESC", conn)
    conn.close()
    df["prediction_date"] = pd.to_datetime(df["prediction_date"])
    df["target_date"] = pd.to_datetime(df["target_date"])
    return df


@st.cache_data(ttl=3600)
def load_eval_report() -> dict:
    """評価レポートを読み込む。"""
    report_path = ARTIFACT_DIR / "eval" / "eval_report.json"
    if not report_path.exists():
        return {}
    with open(report_path) as f:
        return json.load(f)


def render_prediction_chart(predictions: pd.DataFrame) -> None:
    """予測チャート（信頼区間付きファンチャート）。"""
    st.subheader("予測チャート")

    if predictions.empty:
        st.info("予測データがありません。`scripts/predict.py` を実行してください。")
        return

    latest = predictions[predictions["prediction_date"] == predictions["prediction_date"].max()]

    fig = go.Figure()

    # 信頼区間
    fig.add_trace(go.Scatter(
        x=latest["target_date"],
        y=latest["q90"],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=latest["target_date"],
        y=latest["q10"],
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(68, 68, 255, 0.2)",
        name="90% 信頼区間",
    ))

    # 中央値予測
    fig.add_trace(go.Scatter(
        x=latest["target_date"],
        y=latest["median"],
        mode="lines+markers",
        name="中央値予測",
        line=dict(color="blue", width=2),
    ))

    fig.update_layout(
        title="5日先予測（対数リターン）",
        xaxis_title="日付",
        yaxis_title="対数リターン",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_direction_signals(predictions: pd.DataFrame) -> None:
    """方向シグナル表示。"""
    st.subheader("方向シグナル")

    if predictions.empty:
        return

    latest = predictions[predictions["prediction_date"] == predictions["prediction_date"].max()]

    cols = st.columns(PREDICTION_LENGTH)
    for i, (_, row) in enumerate(latest.iterrows()):
        with cols[i]:
            direction = row["direction"]
            color = "green" if direction == "UP" else "red"
            arrow = "^" if direction == "UP" else "v"
            st.metric(
                label=f"{row['horizon']}日先",
                value=f"{arrow} {direction}",
                delta=f"{row['median']:.5f}",
            )


def render_direction_ratio_monitor(predictions: pd.DataFrame) -> None:
    """方向比率モニター。"""
    st.subheader("方向比率モニター")

    report = load_eval_report()
    if not report or "horizons" not in report:
        st.info("評価レポートがありません。`scripts/evaluate.py` を実行してください。")
        return

    horizons = report["horizons"]
    data = []
    for h_name, metrics in horizons.items():
        horizon_num = int(h_name.split("_")[1])
        data.append({
            "horizon": f"{horizon_num}d",
            "実績 UP比率": metrics["actual_up_ratio"],
            "予測 UP比率": metrics["pred_up_ratio"],
        })

    df = pd.DataFrame(data)
    st.bar_chart(df.set_index("horizon"))


def render_event_calendar() -> None:
    """今後のイベントカレンダー。"""
    st.subheader("今後の経済イベント")

    today = date.today()
    events = get_all_event_dates(today.year)
    upcoming = events[events["date"] >= pd.Timestamp(today)]
    upcoming = upcoming.head(15)

    if upcoming.empty:
        st.info("今後のイベントデータがありません。")
        return

    st.dataframe(
        upcoming.rename(columns={"date": "日付", "event_type": "イベント"}),
        hide_index=True,
    )


def render_accuracy_history(predictions: pd.DataFrame) -> None:
    """過去予測の精度推移。"""
    st.subheader("過去予測の精度（直近30日）")

    report = load_eval_report()
    if not report or "horizons" not in report:
        st.info("評価レポートがありません。")
        return

    data = []
    for h_name, metrics in report["horizons"].items():
        horizon_num = int(h_name.split("_")[1])
        data.append({
            "horizon": f"{horizon_num}d",
            "方向精度": metrics["direction_accuracy"],
            "MAE": metrics["mae"],
            "方向比率差": metrics["direction_ratio_gap"],
        })

    st.dataframe(pd.DataFrame(data), hide_index=True)


def render_feature_importance() -> None:
    """TFT の Variable Selection Network による特徴量重要度。"""
    st.subheader("特徴量重要度")

    ckpt_dir = ARTIFACT_DIR / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("*.ckpt")) if ckpt_dir.exists() else []
    if not ckpts:
        st.info("モデルチェックポイントがありません。学習を実行してください。")
        return

    model = TemporalFusionTransformer.load_from_checkpoint(str(ckpts[0]))
    interpretation = model.interpret_output(
        model.predict(
            training.to_dataloader(train=False, batch_size=64, num_workers=0),
            return_x=True,
        ),
    ) if hasattr(model, "interpret_output") else None

    if interpretation is None:
        st.info("特徴量重要度の取得に失敗しました。評価スクリプトで生成されたデータを確認してください。")
        return

    # encoder variable importance
    if "encoder_variables" in interpretation:
        importance = interpretation["encoder_variables"]
        fig = go.Figure(go.Bar(
            x=list(importance.values()),
            y=list(importance.keys()),
            orientation="h",
        ))
        fig.update_layout(title="Encoder 特徴量重要度", height=400)
        st.plotly_chart(fig, use_container_width=True)


def render_attention_heatmap() -> None:
    """TFT の時間方向 Attention ヒートマップ。"""
    st.subheader("Attention ヒートマップ")

    attention_path = ARTIFACT_DIR / "eval" / "attention_weights.npy"
    if not attention_path.exists():
        st.info("Attention データがありません。評価スクリプトで生成してください。")
        return

    import numpy as np
    weights = np.load(str(attention_path))
    # 平均 attention (horizons x encoder_steps)
    avg_weights = weights.mean(axis=0) if weights.ndim > 2 else weights

    fig = go.Figure(go.Heatmap(
        z=avg_weights,
        colorscale="Blues",
    ))
    fig.update_layout(
        title="時間方向 Attention（どの過去日が重要か）",
        xaxis_title="Encoder ステップ (過去)",
        yaxis_title="Decoder ステップ (予測)",
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)


# --- メインレイアウト ---
predictions = load_predictions()

# 上段: 予測チャート + 方向シグナル
render_prediction_chart(predictions)
render_direction_signals(predictions)

# 中段: 方向比率 + イベントカレンダー
col1, col2 = st.columns(2)
with col1:
    render_direction_ratio_monitor(predictions)
with col2:
    render_event_calendar()

# 中下段: 特徴量重要度 + Attention
col3, col4 = st.columns(2)
with col3:
    render_feature_importance()
with col4:
    render_attention_heatmap()

# 下段: 精度履歴
render_accuracy_history(predictions)
```

- [ ] **Step 2: スモークテスト**

```bash
uv run python -c "import dashboard.app"
```

Expected: Streamlit の警告が出るかもしれないが ImportError はなし
（注: Streamlit アプリのインポート時に st.xxx が呼ばれるため、完全なテストは `streamlit run` で行う）

- [ ] **Step 3: コミット**

```bash
git add dashboard/app.py
git commit -m "feat: Streamlit ダッシュボード（予測チャート, 方向比率, イベントカレンダー）"
```

---

### Task 12: Windows タスクスケジューラ設定

**Files:**
- Create: `scripts/schedule_task.bat`

- [ ] **Step 1: scripts/schedule_task.bat を作成**

```bat
@echo off
REM USD/JPY TFT 日次予測タスクのスケジューラ登録
REM 毎営業日 07:00 JST に predict.py を実行する
REM
REM 使い方: 管理者権限でこのバッチファイルを実行
REM   scripts\schedule_task.bat

set TASK_NAME=FX-Speculate-Daily-Predict
set PROJECT_DIR=%~dp0..
set PYTHON_CMD=uv run python scripts/predict.py

echo タスク名: %TASK_NAME%
echo プロジェクト: %PROJECT_DIR%
echo コマンド: %PYTHON_CMD%
echo.

schtasks /create /tn "%TASK_NAME%" /tr "cmd /c cd /d \"%PROJECT_DIR%\" && %PYTHON_CMD%"  /sc daily /st 07:00 /f

if %ERRORLEVEL% == 0 (
    echo.
    echo タスクが正常に登録されました。
    echo 確認: schtasks /query /tn "%TASK_NAME%"
) else (
    echo.
    echo エラー: タスクの登録に失敗しました。管理者権限で実行してください。
)

pause
```

- [ ] **Step 2: コミット**

```bash
git add scripts/schedule_task.bat
git commit -m "feat: Windows タスクスケジューラ設定バッチファイル"
```

---

### Task 13: 統合テスト・最終確認

**Files:**
- Modify: `.gitignore`
- Create: `tests/test_integration.py`

- [ ] **Step 1: 統合テストを作成**

```python
"""統合テスト: パイプライン全体を小さいデータで通す。"""

import numpy as np
import pandas as pd
import pytest
import torch

from data.dataset import prepare_data, split_data, create_datasets
from data.events import compute_event_features
from data.features import (
    compute_technical_features,
    compute_market_returns,
    compute_calendar_features,
)
from model.loss import DirectionAwareQuantileLoss
from model.trainer import build_tft


def _make_synthetic_features(n: int = 500) -> pd.DataFrame:
    """学習可能な合成データを生成する。"""
    np.random.seed(42)
    index = pd.bdate_range("2023-01-02", periods=n)
    close = 150.0 + np.cumsum(np.random.randn(n) * 0.5)

    df = pd.DataFrame({
        "usdjpy_close": close,
        "usdjpy_open": close + np.random.randn(n) * 0.1,
        "usdjpy_high": close + abs(np.random.randn(n) * 0.3),
        "usdjpy_low": close - abs(np.random.randn(n) * 0.3),
        "usdjpy_volume": np.random.randint(1000, 10000, n),
    }, index=index)

    # テクニカル
    ohlcv = df.rename(columns={
        "usdjpy_open": "Open", "usdjpy_high": "High",
        "usdjpy_low": "Low", "usdjpy_close": "Close",
        "usdjpy_volume": "Volume",
    })
    tech = compute_technical_features(ohlcv)

    # リターン
    df["sp500_close"] = close * 30 + np.cumsum(np.random.randn(n))
    df["nikkei_close"] = close * 250 + np.cumsum(np.random.randn(n))
    df["vix_close"] = 20 + np.random.randn(n)
    df["oil_close"] = 70 + np.cumsum(np.random.randn(n) * 0.1)
    df["gold_close"] = 2000 + np.cumsum(np.random.randn(n) * 0.5)
    returns = compute_market_returns(df)

    # マクロ (ラグなし合成データ)
    macro = pd.DataFrame({
        "fred_us_10y": np.full(n, 4.5),
        "fred_jp_10y": np.full(n, 1.0),
        "fred_ff_rate": np.full(n, 5.25),
        "fred_cpi": np.full(n, 310.0),
        "fred_unemployment": np.full(n, 3.7),
        "fred_gdp": np.full(n, 28000.0),
        "fred_m2": np.full(n, 21000.0),
        "fred_dxy": np.full(n, 104.0),
        "rate_diff": np.full(n, 3.5),
    }, index=index)

    # カレンダー + イベント
    calendar = compute_calendar_features(index)
    events = compute_event_features(index)

    all_features = pd.concat([df, tech, returns, macro, calendar, events], axis=1)
    all_features = all_features.loc[:, ~all_features.columns.duplicated()]
    return all_features.dropna()


class TestEndToEndPipeline:
    def test_dataset_creation(self):
        """データセットが正常に作成される。"""
        features = _make_synthetic_features()
        prepped = prepare_data(features)
        train_df, val_df, tune_df, test_df = split_data(prepped)

        assert len(train_df) > 0
        assert len(val_df) > 0

        training, validation = create_datasets(train_df, val_df)
        assert len(training) > 0
        assert len(validation) > 0

    def test_model_forward_pass(self):
        """モデルのフォワードパスが通る。"""
        features = _make_synthetic_features()
        prepped = prepare_data(features)
        train_df, val_df, _, _ = split_data(prepped)
        training, _ = create_datasets(train_df, val_df)

        tft = build_tft(training)
        train_loader = training.to_dataloader(train=True, batch_size=4, num_workers=0)

        batch = next(iter(train_loader))
        x, y = batch
        output = tft(x)
        assert output.prediction.shape[-1] == 5  # quantiles

    def test_loss_backward_pass(self):
        """損失から勾配が流れる。"""
        features = _make_synthetic_features()
        prepped = prepare_data(features)
        train_df, val_df, _, _ = split_data(prepped)
        training, _ = create_datasets(train_df, val_df)

        tft = build_tft(training)
        train_loader = training.to_dataloader(train=True, batch_size=4, num_workers=0)

        batch = next(iter(train_loader))
        x, y = batch
        output = tft(x)
        loss = tft.loss(output.prediction, y)

        if isinstance(loss, dict):
            total_loss = sum(loss.values())
        else:
            total_loss = loss

        total_loss = total_loss.mean()
        total_loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in tft.parameters()
        )
        assert has_grad
```

- [ ] **Step 2: テスト実行**

```bash
uv run pytest tests/test_integration.py -v --timeout=120
```

Expected: ALL PASSED

- [ ] **Step 3: 全テスト実行**

```bash
uv run pytest tests/ -v
```

Expected: ALL PASSED

- [ ] **Step 4: .gitignore 最終確認・コミット**

```bash
git add tests/test_integration.py
git commit -m "feat: 統合テスト（パイプライン全体のエンドツーエンド確認）"
```

---

## 実行順序まとめ

| Task | 内容 | 依存 |
|---|---|---|
| 1 | スキャフォールディング | なし |
| 2 | events.py | Task 1 |
| 3 | fetch.py | Task 1 |
| 4 | features.py | Task 2, 3 |
| 5 | dataset.py | Task 4 |
| 6 | loss.py | Task 1 |
| 7 | trainer.py | Task 5, 6 |
| 8 | train.py | Task 7 |
| 9 | evaluate.py | Task 8 |
| 10 | predict.py | Task 9 |
| 11 | dashboard/app.py | Task 10 |
| 12 | schedule_task.bat | Task 10 |
| 13 | 統合テスト | Task 1-12 |

**並列実行可能:** Task 2 と Task 3 は独立。Task 6 は Task 2-5 と独立。
