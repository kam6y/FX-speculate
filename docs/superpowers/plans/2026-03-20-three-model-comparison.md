# 3モデル比較 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** CatBoost / LightGBM / Transformer の3モデルを独立ノートブックで学習・バックテストし、公平に比較できるシステムを構築する。

**Architecture:** 既存v6ノートブックの共通処理を `scripts/` にモジュール化（data_loader, features, evaluation）し、3つの独立ノートブックからインポートして使う。CatBoost/LightGBMは共通パイプライン、Transformerは独自の時系列ウィンドウパイプラインを持つ。

**Tech Stack:** Python 3.11, uv, CatBoost, LightGBM, PyTorch, Optuna, ta, scikit-learn, SHAP, Pandas, NumPy

**Spec:** `docs/superpowers/specs/2026-03-20-three-model-comparison-design.md`

---

## Task 1: scripts/__init__.py と data_loader.py

**Files:**
- Create: `scripts/__init__.py`
- Create: `scripts/data_loader.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/test_data_loader.py`
- Modify: `pyproject.toml` (pytest追加)

- [ ] **Step 1: テスト基盤とパッケージファイルを作成**

`tests/` ディレクトリ、`scripts/__init__.py`、`tests/__init__.py`、`tests/conftest.py` を作成。
`pyproject.toml` に `pytest` を追加。

```python
# scripts/__init__.py
```

```python
# tests/__init__.py
```

```python
# tests/conftest.py
"""共通テストフィクスチャ"""
import sys
from pathlib import Path

# プロジェクトルートをsys.pathに追加
sys.path.insert(0, str(Path(__file__).parent.parent))
```

```toml
# pyproject.toml に追加:
#   "pytest",  を dependencies に追加
```

- [ ] **Step 2: data_loader.py のテストを書く**

```python
# tests/test_data_loader.py
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
```

- [ ] **Step 3: テストが失敗することを確認**

Run: `cd "C:/Users/daiya/OneDrive/ドキュメント/FX-speculate" && uv run pytest tests/test_data_loader.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 4: data_loader.py を実装**

```python
# scripts/data_loader.py
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

    # timestamp を datetime に変換
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # 重複除去
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df
```

- [ ] **Step 5: テストが通ることを確認**

Run: `cd "C:/Users/daiya/OneDrive/ドキュメント/FX-speculate" && uv run pytest tests/test_data_loader.py -v`
Expected: ALL PASS

- [ ] **Step 6: コミット**

```bash
git add scripts/__init__.py scripts/data_loader.py tests/__init__.py tests/conftest.py tests/test_data_loader.py pyproject.toml
git commit -m "feat: add scripts/data_loader.py with auto-detect parquet loading"
```

---

## Task 2: scripts/features.py

**Files:**
- Create: `scripts/features.py`
- Create: `tests/test_features.py`

v6 Cell 2 の `select_ohlc_columns_v6()` と `generate_features_v6()` を抽出。加えて `create_target()` と `purged_time_series_split()` を Cell 3/5 から抽出。

- [ ] **Step 1: テストを書く**

```python
# tests/test_features.py
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from scripts.data_loader import load_data
from scripts.features import (
    prepare_ohlcv,
    generate_features,
    create_target,
    purged_time_series_split,
    purged_cv_splits,
    get_feature_columns,
    TIME_FEATURES,
)


@pytest.fixture
def sample_df():
    """テスト用に最初の1000行だけ読み込む"""
    df = load_data(str(Path(__file__).parent.parent / "data"))
    return df.head(1000)


def test_prepare_ohlcv(sample_df):
    """OHLCVカラムの準備"""
    df, price_cols = prepare_ohlcv(sample_df)
    assert "close" in price_cols
    assert "open" in price_cols
    assert "high" in price_cols
    assert "low" in price_cols
    assert "volume" in price_cols
    assert isinstance(df.index, pd.DatetimeIndex)


def test_generate_features(sample_df):
    """特徴量生成が100以上のカラムを生成する"""
    df, price_cols = prepare_ohlcv(sample_df)
    df_feat = generate_features(df, price_cols)
    assert len(df_feat.columns) > 100
    # カスタム特徴量の存在確認
    assert "ma_dist_20" in df_feat.columns
    assert "ema_dist_12" in df_feat.columns
    assert "atr_ratio" in df_feat.columns
    assert "bb_width" in df_feat.columns
    assert "return_1" in df_feat.columns
    assert "hour_sin" in df_feat.columns
    assert "is_tokyo_session" in df_feat.columns


def test_get_feature_columns(sample_df):
    """特徴量カラムのフィルタリング（price/target系を除外）"""
    df, price_cols = prepare_ohlcv(sample_df)
    df_feat = generate_features(df, price_cols)
    feat_cols = get_feature_columns(df_feat)
    # ask_, bid_ 系が含まれないこと
    for c in feat_cols:
        assert not c.startswith("ask_")
        assert not c.startswith("bid_")


def test_create_target():
    """ターゲットラベルの生成"""
    close = pd.Series([100.0, 100.1, 99.9, 100.05, 100.0],
                      index=pd.date_range("2024-01-01", periods=5, freq="min"))
    df = pd.DataFrame({"close": close})
    y = create_target(df, threshold_pips=5.0, horizon=1, pip_size=0.01)
    assert set(y.dropna().unique()).issubset({0, 1, 2})


def test_purged_time_series_split(sample_df):
    """Purged分割の比率とギャップ"""
    df, price_cols = prepare_ohlcv(sample_df)
    df_feat = generate_features(df, price_cols)
    train, val, test = purged_time_series_split(df_feat, train_ratio=0.6, val_ratio=0.2, gap_minutes=15)
    total = len(df_feat)
    assert len(train) == pytest.approx(total * 0.6, abs=2)
    # gapによりval/testは若干少なくなる
    assert len(val) < total * 0.2 + 1
    assert len(test) < total * 0.2 + 1
    # train末尾とval先頭に15分以上のギャップ
    gap = val.index[0] - train.index[-1]
    assert gap >= pd.Timedelta(minutes=15)


def test_time_features_constant():
    """TIME_FEATURESが9要素"""
    assert len(TIME_FEATURES) == 9
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `cd "C:/Users/daiya/OneDrive/ドキュメント/FX-speculate" && uv run pytest tests/test_features.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: features.py を実装**

v6 Cell 2/3 のコードを抽出。以下の関数を含む:
- `prepare_ohlcv(df)` → v6の `select_ohlc_columns_v6` に対応
- `generate_features(df, price_cols)` → v6の `generate_features_v6` に対応
- `get_feature_columns(df_feat)` → v6の feature_cols フィルタリングロジック
- `create_target(df, threshold_pips, horizon, pip_size)` → v6のターゲット生成
- `purged_time_series_split(df, train_ratio, val_ratio, gap_minutes)` → v6のPurged分割
- `TIME_FEATURES` 定数

```python
# scripts/features.py
"""共通特徴量生成モジュール

v6ノートブックの特徴量生成・ターゲット作成・データ分割ロジックを共通化。
"""

import numpy as np
import pandas as pd
import ta


TIME_FEATURES = [
    "hour_sin", "hour_cos", "minute_sin", "minute_cos",
    "day_sin", "day_cos",
    "is_tokyo_session", "is_london_session", "is_ny_session",
]


def prepare_ohlcv(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Bid/Ask parquetからOHLCV形式に変換。
    close = (ask_close + bid_close) / 2 (midprice)
    volume = 1.0 (FXデータのためダミー)
    indexをtimestampのDatetimeIndexに設定。

    NOTE: 元のask_*/bid_*カラムはそのまま保持される。
    run_backtest()がBid/Ask価格でエントリー/エグジットを計算するために必要。

    Returns: (df with datetime index, price_cols dict)
    """
    df = df.copy()

    # timestampをindexに設定
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
        df.index.name = None

    # midpriceでOHLCVを作成
    if "ask_close" in df.columns and "bid_close" in df.columns:
        df["close"] = (df["ask_close"] + df["bid_close"]) / 2
        df["open"] = (df["ask_open"] + df["bid_open"]) / 2
        df["high"] = (df["ask_high"] + df["bid_high"]) / 2
        df["low"] = (df["ask_low"] + df["bid_low"]) / 2
    elif "close" not in df.columns:
        raise ValueError("DataFrameにclose価格カラムが見つかりません")

    if "volume" not in df.columns:
        df["volume"] = 1.0

    price_cols = {
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    }
    return df, price_cols


def generate_features(df: pd.DataFrame, price_cols: dict) -> pd.DataFrame:
    """全特徴量を生成して返す。

    1. ta.add_all_ta_features() によるテクニカル指標
    2. カスタム特徴量: MA乖離率, EMA乖離率, ATR比率, BB幅
    3. リターン特徴量: return_1, 5, 15, 60
    4. 時間特徴量: sin/cos, セッションフラグ
    """
    calc_df = df.copy()
    close = calc_df[price_cols["close"]]

    # 1. ta ライブラリ全指標
    calc_df = ta.add_all_ta_features(
        calc_df,
        open=price_cols["open"],
        high=price_cols["high"],
        low=price_cols["low"],
        close=price_cols["close"],
        volume=price_cols["volume"],
        fillna=True,
    )
    calc_df = calc_df.replace([np.inf, -np.inf], np.nan)

    # 2. カスタム特徴量
    # MA乖離率
    for window in [20, 50, 100, 200]:
        sma = close.rolling(window).mean()
        calc_df[f"ma_dist_{window}"] = (close - sma) / sma

    # EMA乖離率
    for window in [12, 26]:
        ema = close.ewm(span=window, adjust=False).mean()
        calc_df[f"ema_dist_{window}"] = (close - ema) / ema

    # ATR比率
    if "volatility_atr" in calc_df.columns:
        atr_ma = calc_df["volatility_atr"].rolling(30).mean()
        calc_df["atr_ratio"] = calc_df["volatility_atr"] / atr_ma

    # ボリンジャーバンド幅
    if "volatility_bbh" in calc_df.columns and "volatility_bbl" in calc_df.columns:
        calc_df["bb_width"] = (calc_df["volatility_bbh"] - calc_df["volatility_bbl"]) / close

    # 3. リターン特徴量
    for lag in [1, 5, 15, 60]:
        calc_df[f"return_{lag}"] = close.pct_change(lag)

    # 4. 時間特徴量
    idx = calc_df.index
    hour = idx.hour
    minute = idx.minute
    day = idx.dayofweek

    calc_df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    calc_df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    calc_df["minute_sin"] = np.sin(2 * np.pi * minute / 60)
    calc_df["minute_cos"] = np.cos(2 * np.pi * minute / 60)
    calc_df["day_sin"] = np.sin(2 * np.pi * day / 7)
    calc_df["day_cos"] = np.cos(2 * np.pi * day / 7)

    calc_df["is_tokyo_session"] = ((hour >= 0) & (hour < 9)).astype(int)
    calc_df["is_london_session"] = ((hour >= 8) & (hour < 17)).astype(int)
    calc_df["is_ny_session"] = ((hour >= 13) & (hour < 22)).astype(int)

    return calc_df


def get_feature_columns(df_feat: pd.DataFrame) -> list[str]:
    """特徴量として使えるカラム名のリストを返す。
    price系・target系・raw系カラムを除外。
    NOTE: "volume"を含むカラム(volume_adi, volume_obv等)も除外される。
    これはv6からの仕様。FXデータではvolumeがダミー値(1.0)のため、
    volume系テクニカル指標は無意味な値になるので除外が正しい。
    """
    exclude_keywords = [
        "ask_", "bid_", "timestamp", "future", "target",
        "tick_", "open", "high", "low", "close", "volume",
    ]
    numeric_cols = df_feat.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if not any(k in c for k in exclude_keywords)]


def create_target(
    df: pd.DataFrame,
    threshold_pips: float,
    horizon: int = 15,
    pip_size: float = 0.01,
) -> pd.Series:
    """Buy(1)/Hold(0)/Sell(2) のターゲットラベルを生成。
    future_return > +threshold_pips → Buy(1)
    future_return < -threshold_pips → Sell(2)
    otherwise → Hold(0)
    """
    close = df["close"] if "close" in df.columns else df.iloc[:, 0]
    future_returns_pips = (close.shift(-horizon) - close) / pip_size
    y = pd.Series(0, index=df.index)
    y[future_returns_pips > threshold_pips] = 1
    y[future_returns_pips < -threshold_pips] = 2
    # horizon末尾はNaNになるのでそのまま（呼び出し側で除外）
    return y


def purged_cv_splits(n_samples: int, n_splits: int, gap: int):
    """Purged Cross-Validation splits。train末尾からgap行を除去してリークを防止。
    Yields: (train_idx, val_idx) のタプル
    """
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, val_idx in tscv.split(np.arange(n_samples)):
        if gap > 0 and len(train_idx) > gap:
            train_idx = train_idx[:-gap]
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        yield train_idx, val_idx


def purged_time_series_split(
    df: pd.DataFrame,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    gap_minutes: int = 15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Purged Time Series Split (60/20/20)。
    train-val間とval-test間にgap_minutesのギャップを挿入。
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    gap = gap_minutes  # 1分足なのでgap_minutes行 = gap_minutes分

    train = df.iloc[:train_end]
    val_start = min(train_end + gap, n)
    val = df.iloc[val_start:val_end]
    test_start = min(val_end + gap, n)
    test = df.iloc[test_start:]

    return train, val, test
```

- [ ] **Step 4: テストが通ることを確認**

Run: `cd "C:/Users/daiya/OneDrive/ドキュメント/FX-speculate" && uv run pytest tests/test_features.py -v`
Expected: ALL PASS

- [ ] **Step 5: コミット**

```bash
git add scripts/features.py tests/test_features.py
git commit -m "feat: add scripts/features.py with shared feature engineering"
```

---

## Task 3: scripts/evaluation.py

**Files:**
- Create: `scripts/evaluation.py`
- Create: `tests/test_evaluation.py`

v6 Cell 5/7 のバックテスト・評価関数を抽出。コストモデルを更新（スプレッド0.2pips, スリッページ0.1pips, API手数料0.002%）。

- [ ] **Step 1: テストを書く**

```python
# tests/test_evaluation.py
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from scripts.evaluation import (
    predict_with_thresholds,
    calc_trade_pnl,
    calculate_sharpe_ratio,
    trade_penalty,
    score_trading,
    build_live_filter,
    run_backtest,
    compute_metrics,
    DEFAULT_CONFIG,
)


def test_predict_with_thresholds():
    """確率からBuy/Hold/Sellを判定"""
    # probs: [hold_prob, buy_prob, sell_prob]
    probs = np.array([
        [0.2, 0.6, 0.2],  # Buy (buy > threshold, buy > sell)
        [0.8, 0.1, 0.1],  # Hold (no prob > threshold)
        [0.2, 0.2, 0.6],  # Sell (sell > threshold, sell > buy)
    ])
    preds = predict_with_thresholds(probs, threshold_buy=0.4, threshold_sell=0.4)
    assert preds[0] == 1  # Buy
    assert preds[1] == 0  # Hold
    assert preds[2] == 2  # Sell


def test_calculate_sharpe_ratio():
    """Sharpe比が正しく計算される"""
    returns = np.array([1.0, 1.0, 1.0, 1.0])
    sharpe = calculate_sharpe_ratio(returns, bar_per_year=362880, trade_rate=0.01)
    assert sharpe > 0  # 全勝なのでプラス


def test_calculate_sharpe_ratio_empty():
    """空リターンで0を返す"""
    assert calculate_sharpe_ratio(np.array([]), 362880, 0.01) == 0.0


def test_trade_penalty():
    """ペナルティが0-1の範囲"""
    penalty = trade_penalty(100, 10000, 50, 50, DEFAULT_CONFIG)
    assert 0.0 <= penalty <= 1.0


def test_build_live_filter():
    """Bad HoursとATRフィルタが機能する"""
    idx = pd.date_range("2024-01-01 20:00", periods=5, freq="h", tz="UTC")
    df_features = pd.DataFrame({"volatility_atr": [0.05, 0.05, 0.05, 0.05, 0.05]}, index=idx)
    config = DEFAULT_CONFIG.copy()
    eligible = build_live_filter(idx, df_features, config, atr_threshold=0.02)
    # UTC 20-23 はBad Hours
    assert not eligible.iloc[0]  # 20:00
    assert not eligible.iloc[1]  # 21:00
    assert not eligible.iloc[2]  # 22:00
    assert not eligible.iloc[3]  # 23:00
    assert eligible.iloc[4]      # 00:00 翌日


def test_default_config_cost_model():
    """コストモデルが設計書の値と一致"""
    assert DEFAULT_CONFIG["SPREAD_PIPS"] == 0.2
    assert DEFAULT_CONFIG["SLIPPAGE_PIPS"] == 0.1
    assert DEFAULT_CONFIG["API_FEE_RATE"] == 0.00002  # 0.002%
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `cd "C:/Users/daiya/OneDrive/ドキュメント/FX-speculate" && uv run pytest tests/test_evaluation.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: evaluation.py を実装**

```python
# scripts/evaluation.py
"""共通バックテスト・評価モジュール

v6のバックテスト・評価ロジックを共通化。
コストモデル: スプレッド0.2pips(Bid/Ask時不要) + スリッページ0.1pips + API手数料0.002%
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_CONFIG = {
    "PREDICT_HORIZON": 15,
    "PIP_SIZE": 0.01,
    "SPREAD_PIPS": 0.2,
    "SLIPPAGE_PIPS": 0.1,
    "API_FEE_RATE": 0.00002,  # 0.002% per side
    "POSITION_SIZE": 10000,
    "BAD_HOURS": [20, 21, 22, 23],
    "ATR_PERCENTILE": 30,
    "EVAL_USE_ATR_FILTER": True,
    "EVAL_USE_BAD_HOURS": True,
    "BAR_PER_YEAR": int(252 * 24 * 60),
    "MIN_TRADES_ABS": 300,
    "MIN_TRADES_RATE": 0.003,
    "MIN_SIDE_TRADES": 20,
}


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity_curve: list[float]
    eligible_bars: int
    use_bid_ask: bool
    total_cost_pips: float


def predict_with_thresholds(
    probs: np.ndarray, threshold_buy: float, threshold_sell: float
) -> np.ndarray:
    """確率配列からBuy(1)/Hold(0)/Sell(2)を判定。"""
    preds = np.zeros(len(probs), dtype=int)
    for i, (hold_p, buy_p, sell_p) in enumerate(probs):
        if buy_p > threshold_buy and buy_p > sell_p:
            preds[i] = 1
        elif sell_p > threshold_sell and sell_p > buy_p:
            preds[i] = 2
    return preds


def calculate_sharpe_ratio(
    returns: np.ndarray, bar_per_year: int, trade_rate: float
) -> float:
    """年率換算Sharpe Ratio。"""
    if len(returns) == 0:
        return 0.0
    r = np.asarray(returns, dtype=float)
    if r.std() == 0:
        return 0.0
    periods_per_year = max(1.0, bar_per_year * trade_rate)
    return float(np.sqrt(periods_per_year) * (r.mean() / r.std()))


def calc_trade_pnl(
    preds: np.ndarray, future_returns_pips: np.ndarray, cost_pips: float
) -> np.ndarray:
    """予測ごとのPnL(pips)を計算。Hold(0)は除外。"""
    pnl = []
    for pred, fr in zip(preds, future_returns_pips):
        if pred == 1:  # Buy
            pnl.append(fr - cost_pips)
        elif pred == 2:  # Sell
            pnl.append(-fr - cost_pips)
    return np.array(pnl) if pnl else np.array([])


def trade_penalty(
    trade_count: int,
    total_count: int,
    buy_count: int,
    sell_count: int,
    config: dict,
) -> float:
    """取引数ペナルティ。最低取引数とBuy/Sellバランスを考慮。"""
    if total_count == 0:
        return 0.0
    min_trades = max(
        config["MIN_TRADES_ABS"],
        int(config["MIN_TRADES_RATE"] * total_count),
    )
    rate_factor = min(1.0, trade_count / max(1, min_trades))
    side_factor = min(
        1.0, min(buy_count, sell_count) / max(1, config["MIN_SIDE_TRADES"])
    )
    balance_factor = min(buy_count, sell_count) / max(
        1, max(buy_count, sell_count)
    )
    return rate_factor * side_factor * (0.5 + 0.5 * balance_factor)


def score_trading(
    preds: np.ndarray,
    future_returns_pips: np.ndarray,
    total_count: int,
    config: dict,
    cost_pips: float,
) -> float:
    """Trading Score = Sharpe * penalty - (1 - penalty)。Optuna目的関数用。"""
    trade_count = int((preds != 0).sum())
    buy_count = int((preds == 1).sum())
    sell_count = int((preds == 2).sum())
    trade_rate = trade_count / total_count if total_count > 0 else 0.0

    pnl_pips = calc_trade_pnl(preds, future_returns_pips, cost_pips)
    sharpe = calculate_sharpe_ratio(pnl_pips, config["BAR_PER_YEAR"], trade_rate)
    penalty = trade_penalty(trade_count, total_count, buy_count, sell_count, config)

    return sharpe * penalty - (1 - penalty)


def build_live_filter(
    index: pd.DatetimeIndex,
    df_features: pd.DataFrame,
    config: dict,
    atr_threshold: float,
) -> pd.Series:
    """Bad HoursとATRフィルタを適用したeligibleマスクを返す。"""
    eligible = pd.Series(True, index=index)
    if config.get("EVAL_USE_BAD_HOURS", True):
        eligible &= ~index.hour.isin(config["BAD_HOURS"])
    if (
        config.get("EVAL_USE_ATR_FILTER", True)
        and "volatility_atr" in df_features.columns
    ):
        eligible &= df_features.loc[index, "volatility_atr"] >= atr_threshold
    return eligible


def run_backtest(
    predictions: np.ndarray,
    timestamps: pd.DatetimeIndex,
    price_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    config: dict,
) -> BacktestResult:
    """Bid/Askバックテスト。独立トレード方式。
    predictions: 0=Hold, 1=Buy, 2=Sell の配列
    config: DEFAULT_CONFIGベース + atr_threshold, threshold_buy, threshold_sell
    """
    use_bid_ask = {"ask_close", "bid_close"}.issubset(price_df.columns)
    slippage_cost = config["SLIPPAGE_PIPS"] * config["PIP_SIZE"]

    if not use_bid_ask:
        spread_cost = config["SPREAD_PIPS"] * config["PIP_SIZE"]
    else:
        spread_cost = 0.0

    atr_threshold = config.get("atr_threshold", 0.0)
    eligible_mask = build_live_filter(timestamps, feature_df, config, atr_threshold)

    results = []
    equity_curve = [0.0]
    eligible_bars = 0

    for i, idx in enumerate(timestamps):
        if not eligible_mask.iloc[i]:
            continue

        exit_idx = idx + pd.Timedelta(minutes=config["PREDICT_HORIZON"])
        if exit_idx not in price_df.index:
            continue

        eligible_bars += 1
        pred = predictions[i]
        if pred == 0:
            continue

        # Entry/Exit price
        if use_bid_ask:
            entry_buy = price_df.loc[idx, "ask_close"]
            exit_buy = price_df.loc[exit_idx, "bid_close"]
            entry_sell = price_df.loc[idx, "bid_close"]
            exit_sell = price_df.loc[exit_idx, "ask_close"]
        else:
            entry_buy = price_df.loc[idx, "close"]
            exit_buy = price_df.loc[exit_idx, "close"]
            entry_sell = entry_buy
            exit_sell = exit_buy

        # API fee (0.002% of notional per side, round trip = 2x)
        api_fee_rate = config.get("API_FEE_RATE", 0.0)

        if pred == 1:  # Buy
            entry_price = entry_buy
            exit_price = exit_buy
            raw_pnl = (exit_price - entry_price) * config["POSITION_SIZE"]
            api_fee = (entry_price + exit_price) * config["POSITION_SIZE"] * api_fee_rate
            pnl = raw_pnl - slippage_cost * config["POSITION_SIZE"] - spread_cost * config["POSITION_SIZE"] - api_fee
            signal = "BUY"
        else:  # Sell
            entry_price = entry_sell
            exit_price = exit_sell
            raw_pnl = (entry_price - exit_price) * config["POSITION_SIZE"]
            api_fee = (entry_price + exit_price) * config["POSITION_SIZE"] * api_fee_rate
            pnl = raw_pnl - slippage_cost * config["POSITION_SIZE"] - spread_cost * config["POSITION_SIZE"] - api_fee
            signal = "SELL"

        results.append({
            "timestamp": idx,
            "signal": signal,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": pnl,
            "win": pnl > 0,
        })
        equity_curve.append(equity_curve[-1] + pnl)

    if not results:
        raise ValueError("No trades generated")

    total_cost_pips = config["SLIPPAGE_PIPS"] + (0 if use_bid_ask else config["SPREAD_PIPS"])

    return BacktestResult(
        trades=pd.DataFrame(results),
        equity_curve=equity_curve,
        eligible_bars=eligible_bars,
        use_bid_ask=use_bid_ask,
        total_cost_pips=total_cost_pips,
    )


def compute_metrics(result: BacktestResult, config: dict = None) -> dict:
    """BacktestResultから全評価指標を計算。"""
    if config is None:
        config = DEFAULT_CONFIG

    df = result.trades
    total_trades = len(df)
    wins = df["win"].sum()
    losses = total_trades - wins
    total_pnl = df["pnl"].sum()
    win_rate = wins / total_trades * 100 if total_trades > 0 else 0
    avg_win = df[df["win"]]["pnl"].mean() if wins > 0 else 0
    avg_loss = df[~df["win"]]["pnl"].mean() if losses > 0 else 0

    gross_profit = df[df["pnl"] > 0]["pnl"].sum()
    gross_loss = df[df["pnl"] < 0]["pnl"].sum()
    profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float("inf")

    equity = pd.Series(result.equity_curve)
    running_max = equity.cummax()
    drawdown = equity - running_max
    max_drawdown = drawdown.min()

    trade_rate = total_trades / result.eligible_bars if result.eligible_bars > 0 else 0
    sharpe = calculate_sharpe_ratio(df["pnl"].values, config["BAR_PER_YEAR"], trade_rate)

    avg_trade_pnl = total_pnl / total_trades if total_trades > 0 else 0
    expected_trades_year = config["BAR_PER_YEAR"] * trade_rate
    annual_return = avg_trade_pnl * expected_trades_year
    calmar = abs(annual_return / max_drawdown) if max_drawdown != 0 else float("inf")

    return {
        "sharpe_ratio": sharpe,
        "total_pnl": total_pnl,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar,
        "total_trades": total_trades,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "trade_rate": trade_rate,
    }


def plot_equity_curve(result: BacktestResult, title: str = "Equity Curve") -> None:
    """エクイティカーブを描画。"""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(result.equity_curve, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Cumulative P&L (JPY)")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
```

- [ ] **Step 4: テストが通ることを確認**

Run: `cd "C:/Users/daiya/OneDrive/ドキュメント/FX-speculate" && uv run pytest tests/test_evaluation.py -v`
Expected: ALL PASS

- [ ] **Step 5: コミット**

```bash
git add scripts/evaluation.py tests/test_evaluation.py
git commit -m "feat: add scripts/evaluation.py with shared backtest and metrics"
```

---

## Task 4: notebooks/usd_jpy_catboost.ipynb (v6リファクタリング)

**Files:**
- Create: `notebooks/usd_jpy_catboost.ipynb`
- Reference: `notebooks/usd_jpy_model_v6.ipynb` (既存)

v6ノートブックを共通モジュールを使う形にリファクタリング。

**NOTE: コストモデルの意図的な変更**
v6の値（SPREAD_PIPS=0.4, SLIPPAGE_PIPS=0.05, API手数料なし）から
設計書の値（SPREAD_PIPS=0.2, SLIPPAGE_PIPS=0.1, API_FEE_RATE=0.002%）に更新。
実データ検証に基づく修正のため、v6とバックテスト結果が異なる。

- [ ] **Step 1: ノートブックを作成**

以下のセル構成で作成:

**Cell 0: Settings & Imports**
```python
"""CatBoost モデル — USD/JPY 1分足 方向予測"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd().parent))

import numpy as np
import pandas as pd
import optuna
import shap
import pickle
import warnings
from catboost import CatBoostClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score, f1_score

from scripts.data_loader import load_data
from scripts.features import (
    prepare_ohlcv, generate_features, get_feature_columns,
    create_target, purged_time_series_split, TIME_FEATURES,
)
from scripts.evaluation import (
    predict_with_thresholds, score_trading, build_live_filter,
    run_backtest, compute_metrics, plot_equity_curve, DEFAULT_CONFIG,
)

USE_GPU = True
ARTIFACT_DIR = Path("../artifacts/catboost")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

def catboost_device_params():
    return {"task_type": "GPU", "devices": "0"} if USE_GPU else {"task_type": "CPU"}

CONFIG = {
    **DEFAULT_CONFIG,
    "THRESHOLD_PIPS_MIN": 2,
    "THRESHOLD_PIPS_MAX": 10,
    "THRESHOLD_PIPS_DEFAULT": 5,
    "TOP_N_FEATURES": 40,
    "RANDOM_SEED": 42,
    "N_TRIALS": 40,
    "CV_SPLITS": 5,
    "TRAIN_RATIO": 0.6,
    "VAL_RATIO": 0.2,
    "PROB_THRESHOLD_MIN": 0.30,
    "PROB_THRESHOLD_MAX": 0.70,
    "TIME_FEATURE_POLICY": "cap",
    "MAX_TIME_FEATURES": 4,
}

np.random.seed(CONFIG["RANDOM_SEED"])
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")
```

**Cell 1: Data Loading & Feature Engineering**
```python
df = load_data(str(Path.cwd().parent / "data"))
df, price_cols = prepare_ohlcv(df)
df_features = generate_features(df, price_cols)
feature_cols = get_feature_columns(df_features)

X_all = df_features[feature_cols].copy()
y_all = create_target(df_features, CONFIG["THRESHOLD_PIPS_DEFAULT"], CONFIG["PREDICT_HORIZON"])

# future_returns_pipsも保持（Optuna目的関数で使用）
future_returns_pips = (df_features["close"].shift(-CONFIG["PREDICT_HORIZON"]) - df_features["close"]) / CONFIG["PIP_SIZE"]

# NaN除去
valid_indices = X_all.dropna().index.intersection(y_all.index)
valid_indices = valid_indices[:-CONFIG["PREDICT_HORIZON"]]
X = X_all.loc[valid_indices]
y = y_all.loc[valid_indices]
future_returns_pips_valid = future_returns_pips.loc[valid_indices]

print(f"Features: {len(feature_cols)}, Samples: {len(X)}")
```

**Cell 2: Purged Train/Val/Test Split**
```python
# 共通モジュールのpurged_time_series_splitを使用
X_train_full, X_val_full, X_test_full = purged_time_series_split(
    X, train_ratio=CONFIG["TRAIN_RATIO"], val_ratio=CONFIG["VAL_RATIO"],
    gap_minutes=CONFIG["PREDICT_HORIZON"],
)

X_train, y_train = X_train_full, y.loc[X_train_full.index]
X_val, y_val = X_val_full, y.loc[X_val_full.index]
X_test, y_test = X_test_full, y.loc[X_test_full.index]
future_returns_pips_train = future_returns_pips_valid.loc[X_train.index]
future_returns_pips_val = future_returns_pips_valid.loc[X_val.index]
future_returns_pips_test = future_returns_pips_valid.loc[X_test.index]

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
```

**Cell 3: Feature Selection (Purged CV)**
```python
from scripts.features import TIME_FEATURES, purged_cv_splits

gap = CONFIG["PREDICT_HORIZON"]  # Purge gap for CV

importance_scores = []
for fold, (tr_idx, va_idx) in enumerate(purged_cv_splits(len(X_train), 3, gap)):
    X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
    X_va, y_va = X_train.iloc[va_idx], y_train.iloc[va_idx]

    classes = np.unique(y_tr)
    weights = compute_class_weight("balanced", classes=classes, y=y_tr)
    cw = dict(zip(classes, weights))

    temp = CatBoostClassifier(
        iterations=300, learning_rate=0.1, depth=6,
        loss_function="MultiClass", class_weights=cw,
        **catboost_device_params(), verbose=0, random_seed=CONFIG["RANDOM_SEED"],
    )
    temp.fit(X_tr, y_tr)
    importance_scores.append(temp.get_feature_importance())

avg_importance = np.mean(importance_scores, axis=0)
df_imp = pd.DataFrame({"feature": X_train.columns, "importance": avg_importance})
df_imp = df_imp.sort_values("importance", ascending=False)

# Time feature capping policy
policy = CONFIG["TIME_FEATURE_POLICY"]
if policy == "cap":
    time_sorted = [f for f in df_imp["feature"] if f in TIME_FEATURES]
    allowed_time = time_sorted[:CONFIG["MAX_TIME_FEATURES"]]
    non_time = [f for f in df_imp["feature"] if f not in TIME_FEATURES]
    selected_features = (allowed_time + non_time)[:CONFIG["TOP_N_FEATURES"]]
elif policy == "drop":
    non_time = df_imp[~df_imp["feature"].isin(TIME_FEATURES)]
    selected_features = non_time.head(CONFIG["TOP_N_FEATURES"])["feature"].tolist()
else:
    selected_features = df_imp.head(CONFIG["TOP_N_FEATURES"])["feature"].tolist()

X_train = X_train[selected_features]
X_val = X_val[selected_features]
X_test = X_test[selected_features]

print(f"Selected {len(selected_features)} features")
print(selected_features[:10])
```

**Cell 4: Optuna Hyperparameter Tuning**
```python
def objective(trial):
    threshold_pips = trial.suggest_float("threshold_pips", CONFIG["THRESHOLD_PIPS_MIN"], CONFIG["THRESHOLD_PIPS_MAX"])
    prob_threshold = trial.suggest_float("prob_threshold", CONFIG["PROB_THRESHOLD_MIN"], CONFIG["PROB_THRESHOLD_MAX"])

    y_opt = create_target(df_features.loc[X_train.index], threshold_pips, CONFIG["PREDICT_HORIZON"])
    y_opt = y_opt.loc[X_train.index]
    classes = np.unique(y_opt)
    if len(classes) < 3:
        return -1.0

    params = {
        "iterations": 1000,
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.12, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 10.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.1, 5.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "loss_function": "MultiClassOneVsAll",
        "eval_metric": "TotalF1",
        **catboost_device_params(),
        "verbose": 0,
        "early_stopping_rounds": 50,
        "random_seed": CONFIG["RANDOM_SEED"],
    }

    cost_pips = CONFIG["SPREAD_PIPS"] + CONFIG["SLIPPAGE_PIPS"]
    scores = []

    for tr_idx, va_idx in purged_cv_splits(len(X_train), CONFIG["CV_SPLITS"], gap):
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr = y_opt.iloc[tr_idx]

        classes_fold = np.unique(y_tr)
        if len(classes_fold) < 2:
            continue
        w = compute_class_weight("balanced", classes=classes_fold, y=y_tr)
        cw = dict(zip(classes_fold, w))

        model = CatBoostClassifier(**params, class_weights=cw)
        y_va_fold = y_opt.iloc[va_idx]
        model.fit(X_tr, y_tr, eval_set=(X_va, y_va_fold))

        probs = model.predict_proba(X_va)
        preds = predict_with_thresholds(probs, prob_threshold, prob_threshold)
        score = score_trading(preds, future_returns_pips_train.iloc[va_idx].values, len(preds), CONFIG, cost_pips)
        scores.append(score)

    return float(np.mean(scores)) if scores else -1.0

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=CONFIG["N_TRIALS"], show_progress_bar=True)

BEST_THRESHOLD_PIPS = study.best_params["threshold_pips"]
BEST_PROB_THRESHOLD = study.best_params["prob_threshold"]
print(f"Best threshold_pips: {BEST_THRESHOLD_PIPS:.4f}")
print(f"Best prob_threshold: {BEST_PROB_THRESHOLD:.4f}")
print(f"Best score: {study.best_value:.4f}")

# ターゲットを最適閾値で再生成
y_all_opt = create_target(df_features, BEST_THRESHOLD_PIPS, CONFIG["PREDICT_HORIZON"])
y_train = y_all_opt.loc[X_train.index]
y_val = y_all_opt.loc[X_val.index]
y_test = y_all_opt.loc[X_test.index]
```

**Cell 5: Train Final Model & Threshold Optimization**
```python
classes = np.unique(y_train)
weights = compute_class_weight("balanced", classes=classes, y=y_train)
class_weights_train = dict(zip(classes, weights))

final_params = {k: v for k, v in study.best_params.items() if k not in ["threshold_pips", "prob_threshold"]}
final_params.update({
    "iterations": 3000,
    "loss_function": "MultiClassOneVsAll",
    "eval_metric": "TotalF1",
    "class_weights": class_weights_train,
    "verbose": 500,
    "early_stopping_rounds": 150,
    "random_seed": CONFIG["RANDOM_SEED"],
})
final_params.update(catboost_device_params())

model = CatBoostClassifier(**final_params)
model.fit(X_train, y_train, eval_set=(X_val, y_val))

# ATR threshold
if "volatility_atr" in df_features.columns:
    atr_series = df_features.loc[X_train.index, "volatility_atr"]
    ATR_THRESHOLD = atr_series.quantile(CONFIG["ATR_PERCENTILE"] / 100)
else:
    ATR_THRESHOLD = 0.0

# Probability threshold grid search on validation
probs_val = model.predict_proba(X_val)
eligible_val = build_live_filter(X_val.index, df_features, CONFIG, ATR_THRESHOLD)
cost_pips = CONFIG["SPREAD_PIPS"] + CONFIG["SLIPPAGE_PIPS"]
best_score = -np.inf
THRESHOLD_BUY, THRESHOLD_SELL = CONFIG["PROB_THRESHOLD_MIN"], CONFIG["PROB_THRESHOLD_MIN"]

for tb in np.arange(CONFIG["PROB_THRESHOLD_MIN"], CONFIG["PROB_THRESHOLD_MAX"] + 1e-9, 0.02):
    for ts in np.arange(CONFIG["PROB_THRESHOLD_MIN"], CONFIG["PROB_THRESHOLD_MAX"] + 1e-9, 0.02):
        preds = predict_with_thresholds(probs_val, tb, ts)
        preds_filtered = preds.copy()
        preds_filtered[~eligible_val.values] = 0
        score = score_trading(preds_filtered, future_returns_pips_val.values, int(eligible_val.sum()), CONFIG, cost_pips)
        if score > best_score:
            best_score = score
            THRESHOLD_BUY, THRESHOLD_SELL = tb, ts

print(f"Optimal thresholds — Buy: {THRESHOLD_BUY:.2f}, Sell: {THRESHOLD_SELL:.2f}")
```

**Cell 6: Backtest on Test Set**
```python
probs_test = model.predict_proba(X_test)
preds_test = predict_with_thresholds(probs_test, THRESHOLD_BUY, THRESHOLD_SELL)

backtest_config = {**CONFIG, "atr_threshold": ATR_THRESHOLD}
result = run_backtest(preds_test, X_test.index, df, df_features, backtest_config)
metrics = compute_metrics(result, CONFIG)

print("=== CatBoost Backtest Results ===")
for k, v in metrics.items():
    if isinstance(v, float):
        print(f"  {k}: {v:.4f}")
    else:
        print(f"  {k}: {v}")

plot_equity_curve(result, title="CatBoost Equity Curve")
```

**Cell 7: Save Artifacts**
```python
model.save_model(str(ARTIFACT_DIR / "model.cbm"))

with open(ARTIFACT_DIR / "selected_features.pkl", "wb") as f:
    pickle.dump(selected_features, f)

config_save = CONFIG.copy()
config_save["THRESHOLD_PIPS"] = BEST_THRESHOLD_PIPS
config_save["THRESHOLD_BUY"] = THRESHOLD_BUY
config_save["THRESHOLD_SELL"] = THRESHOLD_SELL
config_save["ATR_THRESHOLD"] = ATR_THRESHOLD

with open(ARTIFACT_DIR / "config.pkl", "wb") as f:
    pickle.dump(config_save, f)

print(f"Artifacts saved to {ARTIFACT_DIR}")
```

**Cell 8: SHAP Analysis**
```python
sample_size = min(5000, len(X_test))
X_sample = X_test.sample(n=sample_size, random_state=CONFIG["RANDOM_SEED"])
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)
shap.summary_plot(shap_values, X_sample, plot_type="bar", max_display=15)
```

- [ ] **Step 2: ノートブックをCell 0から順に実行して動作確認**

Run: `cd "C:/Users/daiya/OneDrive/ドキュメント/FX-speculate" && uv run jupyter nbconvert --to notebook --execute notebooks/usd_jpy_catboost.ipynb --ExecutePreprocessor.timeout=3600`

注意: GPU使用のため長時間かかる場合がある。エラーが出たらセルごとにデバッグ。

- [ ] **Step 3: コミット**

```bash
git add notebooks/usd_jpy_catboost.ipynb
git commit -m "feat: add CatBoost notebook using shared modules"
```

---

## Task 5: notebooks/usd_jpy_lightgbm.ipynb

**Files:**
- Create: `notebooks/usd_jpy_lightgbm.ipynb`

CatBoostノートブックをベースに、モデル部分のみLightGBMに差し替え。
コストモデルの変更はCatBoostと同一（設計書の値を使用）。

- [ ] **Step 1: ノートブックを作成**

**CatBoostからの差分を以下に記載。Cell 1（データ読み込み）、Cell 2（分割）はCatBoostのCell 1, 2をそのままコピー。Cell 6（バックテスト）はモデル名を"LightGBM"に変更するのみ。Cell 8のSHAP解析もCatBoostと同一（LGBMもTreeExplainer対応）。**

**Cell 0: Settings & Imports**
```python
"""LightGBM モデル — USD/JPY 1分足 方向予測"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd().parent))

import numpy as np
import pandas as pd
import optuna
import shap
import pickle
import warnings
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score, f1_score

from scripts.data_loader import load_data
from scripts.features import (
    prepare_ohlcv, generate_features, get_feature_columns,
    create_target, purged_time_series_split, TIME_FEATURES,
)
from scripts.evaluation import (
    predict_with_thresholds, score_trading, build_live_filter,
    run_backtest, compute_metrics, plot_equity_curve, DEFAULT_CONFIG,
)

ARTIFACT_DIR = Path("../artifacts/lightgbm")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = {
    **DEFAULT_CONFIG,
    "THRESHOLD_PIPS_MIN": 2,
    "THRESHOLD_PIPS_MAX": 10,
    "THRESHOLD_PIPS_DEFAULT": 5,
    "TOP_N_FEATURES": 40,
    "RANDOM_SEED": 42,
    "N_TRIALS": 40,
    "CV_SPLITS": 5,
    "TRAIN_RATIO": 0.6,
    "VAL_RATIO": 0.2,
    "PROB_THRESHOLD_MIN": 0.30,
    "PROB_THRESHOLD_MAX": 0.70,
    "TIME_FEATURE_POLICY": "cap",
    "MAX_TIME_FEATURES": 4,
}

np.random.seed(CONFIG["RANDOM_SEED"])
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")
```

**Cell 1: Data Loading & Feature Engineering — CatBoost Cell 1と同一コード（そのままコピー）**

**Cell 2: Purged Train/Val/Test Split — CatBoost Cell 2と同一コード（そのままコピー）**

**Cell 3: Feature Selection — LightGBM版**
Feature selectionのみ、CatBoostの代わりにLightGBMで特徴量重要度を計算:

```python
# Cell 3: Feature Selection — LightGBM版
from scripts.features import TIME_FEATURES, purged_cv_splits

gap = CONFIG["PREDICT_HORIZON"]  # Purge gap for CV

importance_scores = []
for fold, (tr_idx, va_idx) in enumerate(purged_cv_splits(len(X_train), 3, gap)):
    X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]

    classes = np.unique(y_tr)
    weights = compute_class_weight("balanced", classes=classes, y=y_tr)
    sample_weights = np.array([weights[list(classes).index(c)] for c in y_tr])

    temp = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.1, max_depth=6,
        objective="multiclass", num_class=3,
        random_state=CONFIG["RANDOM_SEED"], verbose=-1,
    )
    temp.fit(X_tr, y_tr, sample_weight=sample_weights)
    importance_scores.append(temp.feature_importances_)

# 以降の feature selection ロジックはCatBoostと同一
```

**Cell 4: Optuna — LightGBM版**
```python
def objective(trial):
    threshold_pips = trial.suggest_float("threshold_pips", CONFIG["THRESHOLD_PIPS_MIN"], CONFIG["THRESHOLD_PIPS_MAX"])
    prob_threshold = trial.suggest_float("prob_threshold", CONFIG["PROB_THRESHOLD_MIN"], CONFIG["PROB_THRESHOLD_MAX"])

    y_opt = create_target(df_features.loc[X_train.index], threshold_pips, CONFIG["PREDICT_HORIZON"])
    y_opt = y_opt.loc[X_train.index]
    classes = np.unique(y_opt)
    if len(classes) < 3:
        return -1.0

    params = {
        "n_estimators": 1000,
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.12, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "objective": "multiclass",
        "num_class": 3,
        "random_state": CONFIG["RANDOM_SEED"],
        "verbose": -1,
    }

    cost_pips = CONFIG["SPREAD_PIPS"] + CONFIG["SLIPPAGE_PIPS"]
    scores = []

    for tr_idx, va_idx in purged_cv_splits(len(X_train), CONFIG["CV_SPLITS"], gap):
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr = y_opt.iloc[tr_idx]
        y_va = y_opt.iloc[va_idx]

        classes_fold = np.unique(y_tr)
        if len(classes_fold) < 2:
            continue
        w = compute_class_weight("balanced", classes=classes_fold, y=y_tr)
        sample_weights = np.array([w[list(classes_fold).index(c)] for c in y_tr])

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr, sample_weight=sample_weights,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

        probs = model.predict_proba(X_va)
        preds = predict_with_thresholds(probs, prob_threshold, prob_threshold)
        score = score_trading(preds, future_returns_pips_train.iloc[va_idx].values, len(preds), CONFIG, cost_pips)
        scores.append(score)

    return float(np.mean(scores)) if scores else -1.0

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=CONFIG["N_TRIALS"], show_progress_bar=True)
```

**Cell 5: Final Model — LightGBM版**
```python
final_params = {k: v for k, v in study.best_params.items() if k not in ["threshold_pips", "prob_threshold"]}
final_params.update({
    "n_estimators": 3000,
    "objective": "multiclass",
    "num_class": 3,
    "random_state": CONFIG["RANDOM_SEED"],
    "verbose": -1,
})

classes = np.unique(y_train)
weights = compute_class_weight("balanced", classes=classes, y=y_train)
sample_weights_train = np.array([weights[list(classes).index(c)] for c in y_train])

model = lgb.LGBMClassifier(**final_params)
model.fit(
    X_train, y_train, sample_weight=sample_weights_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(500)],
)
```

**Cell 5 continued: ATR Threshold & Probability Threshold Optimization — CatBoost Cell 5の後半と同一ロジック（そのままコピー）**

**Cell 6: Backtest on Test Set — CatBoost Cell 6と同一。タイトルのみ変更:**
```python
print("=== LightGBM Backtest Results ===")
# ... 以降同一
plot_equity_curve(result, title="LightGBM Equity Curve")
```

**Cell 7: Save Artifacts — LightGBM版**
```python
model.booster_.save_model(str(ARTIFACT_DIR / "model.txt"))

with open(ARTIFACT_DIR / "selected_features.pkl", "wb") as f:
    pickle.dump(selected_features, f)

config_save = CONFIG.copy()
config_save["THRESHOLD_PIPS"] = BEST_THRESHOLD_PIPS
config_save["THRESHOLD_BUY"] = THRESHOLD_BUY
config_save["THRESHOLD_SELL"] = THRESHOLD_SELL
config_save["ATR_THRESHOLD"] = ATR_THRESHOLD

with open(ARTIFACT_DIR / "config.pkl", "wb") as f:
    pickle.dump(config_save, f)

print(f"Artifacts saved to {ARTIFACT_DIR}")
```

**Cell 8: SHAP Analysis — CatBoost Cell 8と同一（LightGBMもTreeExplainer対応）**

- [ ] **Step 2: ノートブックを実行して動作確認**

Run: `cd "C:/Users/daiya/OneDrive/ドキュメント/FX-speculate" && uv run jupyter nbconvert --to notebook --execute notebooks/usd_jpy_lightgbm.ipynb --ExecutePreprocessor.timeout=3600`

- [ ] **Step 3: コミット**

```bash
git add notebooks/usd_jpy_lightgbm.ipynb
git commit -m "feat: add LightGBM notebook using shared modules"
```

---

## Task 6: PyTorch 依存関係の追加

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: pyproject.toml に torch を追加**

```toml
[project]
dependencies = [
  # ... existing deps ...
  "torch>=2.0,<3.0",
]

[tool.uv]
extra-index-url = ["https://download.pytorch.org/whl/cu121"]
```

- [ ] **Step 2: uv lock & sync で依存関係を解決**

Run: `cd "C:/Users/daiya/OneDrive/ドキュメント/FX-speculate" && uv lock && uv sync`

- [ ] **Step 3: PyTorchのインポート確認**

Run: `cd "C:/Users/daiya/OneDrive/ドキュメント/FX-speculate" && uv run python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"`

- [ ] **Step 4: コミット**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add PyTorch dependency for Transformer model"
```

---

## Task 7: notebooks/usd_jpy_transformer.ipynb

**Files:**
- Create: `notebooks/usd_jpy_transformer.ipynb`

独自パイプラインのTransformerエンコーダモデル。

- [ ] **Step 1: ノートブックを作成**

**Cell 0: Settings & Imports**
```python
"""Transformer Encoder モデル — USD/JPY 1分足 方向予測"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd().parent))

import numpy as np
import pandas as pd
import optuna
import pickle
import warnings
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight

from scripts.data_loader import load_data
from scripts.features import (
    prepare_ohlcv, generate_features, get_feature_columns,
    create_target, TIME_FEATURES,
)
from scripts.evaluation import (
    predict_with_thresholds, score_trading, build_live_filter,
    run_backtest, compute_metrics, plot_equity_curve, DEFAULT_CONFIG,
)

ARTIFACT_DIR = Path("../artifacts/transformer")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

CONFIG = {
    **DEFAULT_CONFIG,
    "THRESHOLD_PIPS_MIN": 2,
    "THRESHOLD_PIPS_MAX": 10,
    "THRESHOLD_PIPS_DEFAULT": 5,
    "RANDOM_SEED": 42,
    "N_TRIALS": 20,  # Transformer は学習コストが高いため少なめ
    "TRAIN_RATIO": 0.6,
    "VAL_RATIO": 0.2,
    "PROB_THRESHOLD_MIN": 0.30,
    "PROB_THRESHOLD_MAX": 0.70,
}

np.random.seed(CONFIG["RANDOM_SEED"])
torch.manual_seed(CONFIG["RANDOM_SEED"])
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")
```

**Cell 1: Data Loading & Feature Engineering**
```python
df = load_data(str(Path.cwd().parent / "data"))
df, price_cols = prepare_ohlcv(df)
df_features = generate_features(df, price_cols)

# Transformer用: 全特徴量（raw OHLC + テクニカル指標）を使用。
# CatBoost/LightGBMではTop40選択するが、Transformerでは特徴量選択をスキップ。
# 理由: Transformerは時系列ウィンドウ入力で、Self-Attentionにより自動的に
# 重要な特徴量に注目できるため。また、ウィンドウ単位のStandardScalerにより
# 生価格の非定常性も正規化される。
feature_cols = get_feature_columns(df_features)
# OHLCも追加（Transformerは生価格パターンも学習に使う）
for col in ["open", "high", "low", "close"]:
    if col not in feature_cols:
        feature_cols.append(col)

future_returns_pips = (df_features["close"].shift(-CONFIG["PREDICT_HORIZON"]) - df_features["close"]) / CONFIG["PIP_SIZE"]

# NaN除去
X_all = df_features[feature_cols].copy()
valid_indices = X_all.dropna().index
valid_indices = valid_indices[:-CONFIG["PREDICT_HORIZON"]]
X_all = X_all.loc[valid_indices]
future_returns_pips_valid = future_returns_pips.loc[valid_indices]

print(f"Features: {len(feature_cols)}, Samples: {len(X_all)}")
```

**Cell 2: Purged Split (with extended gap for windowing)**
```python
def purged_split_for_transformer(X, gap_base, max_window_size):
    """Transformer用のPurged分割。gap = gap_base + max_window_size。"""
    n = len(X)
    train_end = int(n * CONFIG["TRAIN_RATIO"])
    val_end = int(n * (CONFIG["TRAIN_RATIO"] + CONFIG["VAL_RATIO"]))
    gap = gap_base + max_window_size

    train = X.iloc[:train_end]
    val_start = min(train_end + gap, n)
    val = X.iloc[val_start:val_end]
    test_start = min(val_end + gap, n)
    test = X.iloc[test_start:]

    return train, val, test

# 最大ウィンドウサイズ（Optunaの上限）でsplitし、小さいウィンドウでも安全
MAX_WINDOW = 180
X_train_raw, X_val_raw, X_test_raw = purged_split_for_transformer(
    X_all, CONFIG["PREDICT_HORIZON"], MAX_WINDOW
)
fr_train = future_returns_pips_valid.loc[X_train_raw.index]
fr_val = future_returns_pips_valid.loc[X_val_raw.index]
fr_test = future_returns_pips_valid.loc[X_test_raw.index]

print(f"Train: {len(X_train_raw)}, Val: {len(X_val_raw)}, Test: {len(X_test_raw)}")
```

**Cell 3: Dataset & Model Definition**
```python
class TimeSeriesDataset(Dataset):
    """ウィンドウ化された時系列データセット。ウィンドウ単位で標準化。"""

    def __init__(self, X: pd.DataFrame, y: pd.Series, window_size: int):
        self.X = X.values.astype(np.float32)
        self.y = y.values.astype(np.int64)
        self.window_size = window_size
        self.valid_indices = list(range(window_size, len(X)))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        end = self.valid_indices[idx]
        start = end - self.window_size
        window = self.X[start:end].copy()

        # ウィンドウ単位で標準化
        mean = window.mean(axis=0, keepdims=True)
        std = window.std(axis=0, keepdims=True) + 1e-8
        window = (window - mean) / std

        return torch.tensor(window), torch.tensor(self.y[end])


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class FXTransformer(nn.Module):
    def __init__(self, n_features, d_model, nhead, num_layers, dim_feedforward, dropout, num_classes=3):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = x[:, -1, :]  # 最終時点のみ使用
        x = self.dropout(x)
        return self.classifier(x)
```

**Cell 4: Training Function**
```python
def train_transformer(
    X_train, y_train, X_val, y_val,
    window_size, d_model, nhead, num_layers, dim_feedforward,
    dropout, lr, batch_size, max_epochs, patience=10,
):
    """Transformerモデルの学習。早期停止あり。"""
    train_ds = TimeSeriesDataset(X_train, y_train, window_size)
    val_ds = TimeSeriesDataset(X_val, y_val, window_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    n_features = X_train.shape[1]
    model = FXTransformer(n_features, d_model, nhead, num_layers, dim_feedforward, dropout).to(DEVICE)

    # Class weights
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                output = model(X_batch)
                val_loss += criterion(output, y_batch).item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state)
    return model


def predict_transformer(model, X, y_dummy, window_size, batch_size=512):
    """Transformerモデルで予測確率を取得。"""
    ds = TimeSeriesDataset(X, y_dummy, window_size)
    loader = DataLoader(ds, batch_size=batch_size)
    model.eval()
    all_probs = []
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(DEVICE)
            output = model(X_batch)
            probs = torch.softmax(output, dim=1).cpu().numpy()
            all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)
```

**Cell 5: Optuna Hyperparameter Tuning**
```python
def objective(trial):
    threshold_pips = trial.suggest_float("threshold_pips", CONFIG["THRESHOLD_PIPS_MIN"], CONFIG["THRESHOLD_PIPS_MAX"])
    prob_threshold = trial.suggest_float("prob_threshold", CONFIG["PROB_THRESHOLD_MIN"], CONFIG["PROB_THRESHOLD_MAX"])
    window_size = trial.suggest_int("window_size", 30, 180, step=10)

    y_train = create_target(df_features.loc[X_train_raw.index], threshold_pips, CONFIG["PREDICT_HORIZON"])
    y_val = create_target(df_features.loc[X_val_raw.index], threshold_pips, CONFIG["PREDICT_HORIZON"])

    if len(np.unique(y_train)) < 3:
        return -1.0

    d_model = trial.suggest_categorical("d_model", [32, 64, 128])
    nhead = trial.suggest_categorical("nhead", [2, 4, 8])
    # d_model must be divisible by nhead
    if d_model % nhead != 0:
        return -1.0

    num_layers = trial.suggest_int("num_layers", 1, 4)
    dim_feedforward = trial.suggest_categorical("dim_feedforward", [64, 128, 256])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

    try:
        model = train_transformer(
            X_train_raw, y_train.loc[X_train_raw.index],
            X_val_raw, y_val.loc[X_val_raw.index],
            window_size, d_model, nhead, num_layers, dim_feedforward,
            dropout, lr, batch_size, max_epochs=50, patience=10,
        )

        probs = predict_transformer(model, X_val_raw, y_val.loc[X_val_raw.index], window_size)
        # probs は window_size 以降のインデックスに対応
        val_indices = X_val_raw.index[window_size:]
        preds = predict_with_thresholds(probs, prob_threshold, prob_threshold)

        cost_pips = CONFIG["SPREAD_PIPS"] + CONFIG["SLIPPAGE_PIPS"]
        fr_val_aligned = fr_val.loc[val_indices].values
        score = score_trading(preds, fr_val_aligned, len(preds), CONFIG, cost_pips)
        return score
    except Exception as e:
        print(f"Trial failed: {e}")
        return -1.0

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=CONFIG["N_TRIALS"], show_progress_bar=True)

print(f"Best score: {study.best_value:.4f}")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")
```

**Cell 6: Train Final Model**
```python
bp = study.best_params
BEST_THRESHOLD_PIPS = bp["threshold_pips"]
BEST_PROB_THRESHOLD = bp["prob_threshold"]
BEST_WINDOW_SIZE = bp["window_size"]

y_train_final = create_target(df_features.loc[X_train_raw.index], BEST_THRESHOLD_PIPS, CONFIG["PREDICT_HORIZON"])
y_val_final = create_target(df_features.loc[X_val_raw.index], BEST_THRESHOLD_PIPS, CONFIG["PREDICT_HORIZON"])
y_test_final = create_target(df_features.loc[X_test_raw.index], BEST_THRESHOLD_PIPS, CONFIG["PREDICT_HORIZON"])

model = train_transformer(
    X_train_raw, y_train_final.loc[X_train_raw.index],
    X_val_raw, y_val_final.loc[X_val_raw.index],
    BEST_WINDOW_SIZE, bp["d_model"], bp["nhead"], bp["num_layers"],
    bp["dim_feedforward"], bp["dropout"], bp["lr"], bp["batch_size"],
    max_epochs=200, patience=20,
)

# ATR threshold
if "volatility_atr" in df_features.columns:
    atr_series = df_features.loc[X_train_raw.index, "volatility_atr"]
    ATR_THRESHOLD = atr_series.quantile(CONFIG["ATR_PERCENTILE"] / 100)
else:
    ATR_THRESHOLD = 0.0

# Threshold optimization on validation
probs_val = predict_transformer(model, X_val_raw, y_val_final.loc[X_val_raw.index], BEST_WINDOW_SIZE)
val_indices = X_val_raw.index[BEST_WINDOW_SIZE:]
eligible_val = build_live_filter(val_indices, df_features, CONFIG, ATR_THRESHOLD)
cost_pips = CONFIG["SPREAD_PIPS"] + CONFIG["SLIPPAGE_PIPS"]

best_score = -np.inf
THRESHOLD_BUY, THRESHOLD_SELL = CONFIG["PROB_THRESHOLD_MIN"], CONFIG["PROB_THRESHOLD_MIN"]
for tb in np.arange(CONFIG["PROB_THRESHOLD_MIN"], CONFIG["PROB_THRESHOLD_MAX"] + 1e-9, 0.02):
    for ts in np.arange(CONFIG["PROB_THRESHOLD_MIN"], CONFIG["PROB_THRESHOLD_MAX"] + 1e-9, 0.02):
        preds = predict_with_thresholds(probs_val, tb, ts)
        preds_f = preds.copy()
        preds_f[~eligible_val.values] = 0
        score = score_trading(preds_f, fr_val.loc[val_indices].values, int(eligible_val.sum()), CONFIG, cost_pips)
        if score > best_score:
            best_score = score
            THRESHOLD_BUY, THRESHOLD_SELL = tb, ts

print(f"Optimal thresholds — Buy: {THRESHOLD_BUY:.2f}, Sell: {THRESHOLD_SELL:.2f}")
```

**Cell 7: Backtest on Test Set**
```python
probs_test = predict_transformer(model, X_test_raw, y_test_final.loc[X_test_raw.index], BEST_WINDOW_SIZE)
test_indices = X_test_raw.index[BEST_WINDOW_SIZE:]
preds_test = predict_with_thresholds(probs_test, THRESHOLD_BUY, THRESHOLD_SELL)

backtest_config = {**CONFIG, "atr_threshold": ATR_THRESHOLD}
result = run_backtest(preds_test, test_indices, df, df_features, backtest_config)
metrics = compute_metrics(result, CONFIG)

print("=== Transformer Backtest Results ===")
for k, v in metrics.items():
    if isinstance(v, float):
        print(f"  {k}: {v:.4f}")
    else:
        print(f"  {k}: {v}")

plot_equity_curve(result, title="Transformer Equity Curve")
```

**Cell 8: Save Artifacts**
```python
torch.save(model.state_dict(), ARTIFACT_DIR / "model.pt")

config_save = CONFIG.copy()
config_save["THRESHOLD_PIPS"] = BEST_THRESHOLD_PIPS
config_save["THRESHOLD_BUY"] = THRESHOLD_BUY
config_save["THRESHOLD_SELL"] = THRESHOLD_SELL
config_save["ATR_THRESHOLD"] = ATR_THRESHOLD
config_save["WINDOW_SIZE"] = BEST_WINDOW_SIZE
config_save["model_params"] = {
    "n_features": len(feature_cols),
    "d_model": bp["d_model"],
    "nhead": bp["nhead"],
    "num_layers": bp["num_layers"],
    "dim_feedforward": bp["dim_feedforward"],
    "dropout": bp["dropout"],
}
config_save["feature_cols"] = feature_cols

with open(ARTIFACT_DIR / "config.pkl", "wb") as f:
    pickle.dump(config_save, f)

print(f"Artifacts saved to {ARTIFACT_DIR}")
```

- [ ] **Step 2: ノートブックを実行して動作確認**

Run: `cd "C:/Users/daiya/OneDrive/ドキュメント/FX-speculate" && uv run jupyter nbconvert --to notebook --execute notebooks/usd_jpy_transformer.ipynb --ExecutePreprocessor.timeout=7200`

注意: Transformer学習は長時間かかる。タイムアウトを2時間に設定。

- [ ] **Step 3: コミット**

```bash
git add notebooks/usd_jpy_transformer.ipynb
git commit -m "feat: add Transformer notebook with time-series window pipeline"
```

---

## Task 8: 旧ノートブックの整理 & 全テスト実行

**Files:**
- Delete or move: `notebooks/usd_jpy_model_v6.ipynb` (旧)

- [ ] **Step 1: 全共通モジュールテストを実行**

Run: `cd "C:/Users/daiya/OneDrive/ドキュメント/FX-speculate" && uv run pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 2: 旧v6ノートブックを削除（gitで追跡されている場合）**

ユーザーに確認してから実施。

```bash
git rm notebooks/usd_jpy_model_v6.ipynb
git commit -m "chore: remove legacy v6 notebook (replaced by model-specific notebooks)"
```

- [ ] **Step 3: 最終確認とコミット**

```bash
git status
# 意図しないファイル（.env, artifacts/等）が含まれていないことを確認してから:
git add scripts/ tests/ notebooks/ docs/ pyproject.toml uv.lock
git commit -m "feat: complete 3-model comparison system (CatBoost, LightGBM, Transformer)"
```
