"""
TFT USD/JPY 予測AI -Temporal Fusion Transformerベース為替予測システム

設計書: TFT_USDJPY予測AI_設計書.md に基づく実装
PyTorch Forecasting + PyTorch Lightning による多段階予測 + 分位点予測パイプライン

必要な追加パッケージ:
    uv add pytorch-forecasting lightning yfinance
    # FRED APIを使う場合: uv add fredapi

実行:
    uv run python train_tft.py                # 通常学習
    uv run python train_tft.py --optuna       # Optunaハイパラチューニング付き
    uv run python train_tft.py --walkforward  # Walk-Forwardバックテスト
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import shutil
import tempfile

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")  # プロジェクトルートの .env を読み込み

# curl-cffi は非ASCII パスのCA証明書を読めないため、安全なパスにコピー
try:
    import certifi

    _cert = certifi.where()
    if not _cert.isascii():
        _safe = str(Path(tempfile.gettempdir()) / "cacert.pem")
        shutil.copy2(_cert, _safe)
        os.environ["CURL_CA_BUNDLE"] = _safe
except ImportError:
    pass

import numpy as np
import pandas as pd
import ta
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer, EncoderNormalizer  # R3-01
from pytorch_forecasting.metrics import QuantileLoss
import optuna


class DirectionAwareQuantileLoss(QuantileLoss):
    """QuantileLoss に方向ペナルティを加えたカスタム損失。

    median 予測の符号が実績と一致しない場合に追加ペナルティを課す。
    これにより TFT が「0に近い安全な予測」ではなく方向性を学習する。
    """

    def __init__(self, quantiles=None, direction_weight: float = 1.0, **kwargs):
        super().__init__(quantiles=quantiles, **kwargs)
        self.direction_weight = direction_weight

    def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 標準 quantile loss
        ql = super().loss(y_pred, target)

        # R8-04: 符号ベース方向ペナルティ
        q_mid = y_pred.size(-1) // 2
        pred_median = y_pred[..., q_mid]
        sign_mismatch = (pred_median * target < 0).float()
        dir_penalty = sign_mismatch * (pred_median - target).abs()

        return ql + self.direction_weight * dir_penalty.unsqueeze(-1)

# ===================================================================
# GPU 最適化: Tensor Core (TF32) + cuDNN autotune
# ===================================================================
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("medium")  # TF32 で ~2x 高速化
    torch.backends.cudnn.benchmark = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = DEVICE.type == "cuda"

try:
    import yfinance as yf

    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

try:
    from fredapi import Fred

    HAS_FRED = True
except ImportError:
    HAS_FRED = False

try:
    import holidays as _holidays_lib

    HAS_HOLIDAYS = True
except ImportError:
    HAS_HOLIDAYS = False


# ===================================================================
# CONFIG -設計書 Section 2.2 / 4.3 に対応
# ===================================================================
CONFIG = {
    # --- データ収集 ---
    "START_DATE": "2019-01-01",
    "END_DATE": None,  # None = 本日
    # --- TFT アーキテクチャ (Section 2.2) ---
    "HIDDEN_SIZE": 192,
    "ATTENTION_HEAD_SIZE": 2,
    "DROPOUT": 0.15,
    "HIDDEN_CONTINUOUS_SIZE": 8,
    "MAX_ENCODER_LENGTH": 120,  # R6-10: 90→120
    "MAX_PREDICTION_LENGTH": 10,
    "QUANTILES": [0.1, 0.5, 0.9],
    # --- 学習 (Section 4.3) ---
    "LEARNING_RATE": 1e-3,
    "BATCH_SIZE": 64,  # R6-07: 32→64 (accum=2なので実効128)
    "MAX_EPOCHS": 50,
    "PATIENCE": 8,
    "GRADIENT_CLIP_VAL": 1.0,
    "DIRECTION_LOSS_WEIGHT": 0.5,
    # --- データ分割 (Section 3.2) ---
    "TRAIN_RATIO": 0.70,
    "VAL_RATIO": 0.15,
    # --- Optuna (Section 4.3) ---
    "N_TRIALS": 15,
    "OPTUNA_TIMEOUT": 3600,
    # --- Walk-Forward (Section 5.2) ---
    "WF_INITIAL_TRAIN_DAYS": 750,
    "WF_TEST_WINDOW": 20,
    "WF_FINETUNE_EPOCHS": 20,
    # --- トレーディング ---
    "SPREAD_PIPS": 0.3,
    "SLIPPAGE_PIPS": 0.1,
    "PIP_SIZE": 0.01,
    "DIRECTION_THRESHOLD_PIPS": 0,
    # --- その他 ---
    "FRED_API_KEY": os.environ.get("FRED_API_KEY"),
    "RANDOM_SEED": 42,
    "ENSEMBLE_SEEDS": 3,
}

BASE_DIR = Path(__file__).parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


# ===================================================================
# FOMC / BOJ 会合日定数 (決定発表日)
# ===================================================================
FOMC_DATES = [
    # 2019
    "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19",
    "2019-07-31", "2019-09-18", "2019-10-30", "2019-12-11",
    # 2020
    "2020-01-29", "2020-03-03", "2020-03-15", "2020-04-29",
    "2020-06-10", "2020-07-29", "2020-09-16", "2020-11-05", "2020-12-16",
    # 2021
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16",
    "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
    # 2022
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15",
    "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    # 2023
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
    "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-17",
    # 2026
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-11-04", "2026-12-16",
]

BOJ_DATES = [
    # 2019
    "2019-01-23", "2019-03-15", "2019-04-25", "2019-06-20",
    "2019-07-30", "2019-09-19", "2019-10-31", "2019-12-19",
    # 2020
    "2020-01-21", "2020-03-16", "2020-04-27", "2020-06-16",
    "2020-07-15", "2020-09-17", "2020-10-29", "2020-12-18",
    # 2021
    "2021-01-21", "2021-03-19", "2021-04-27", "2021-06-18",
    "2021-07-16", "2021-09-22", "2021-10-28", "2021-12-17",
    # 2022
    "2022-01-18", "2022-03-18", "2022-04-28", "2022-06-17",
    "2022-07-21", "2022-09-22", "2022-10-28", "2022-12-20",
    # 2023
    "2023-01-18", "2023-03-10", "2023-04-28", "2023-06-16",
    "2023-07-28", "2023-09-22", "2023-10-31", "2023-12-19",
    # 2024
    "2024-01-23", "2024-03-19", "2024-04-26", "2024-06-14",
    "2024-07-31", "2024-09-20", "2024-10-31", "2024-12-19",
    # 2025
    "2025-01-24", "2025-03-14", "2025-05-01", "2025-06-17",
    "2025-07-31", "2025-09-19", "2025-10-31", "2025-12-19",
    # 2026
    "2026-01-22", "2026-03-13", "2026-04-28", "2026-06-16",
    "2026-07-16", "2026-09-17", "2026-10-29", "2026-12-18",
]


# ===================================================================
# 1. データ収集 (Section 3)
# ===================================================================
def fetch_market_data(config: dict) -> pd.DataFrame:
    """Yahoo Finance から USD/JPY, VIX, Gold, US10Y の日次データを取得"""
    if not HAS_YFINANCE:
        raise ImportError("yfinance が必要です: uv add yfinance")

    end = config["END_DATE"] or pd.Timestamp.now().strftime("%Y-%m-%d")
    start = config["START_DATE"]
    print(f"データ取得: {start} ~ {end}")

    tickers = ["JPY=X", "^VIX", "GC=F", "^TNX"]
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False, group_by="ticker")

    fx = raw["JPY=X"]
    df = pd.DataFrame(
        {
            "open": fx["Open"].squeeze(),
            "high": fx["High"].squeeze(),
            "low": fx["Low"].squeeze(),
            "close": fx["Close"].squeeze(),
        }
    )

    for ticker, col_name in [("^VIX", "vix"), ("GC=F", "gold"), ("^TNX", "us10y")]:
        data = raw[ticker]
        if not data["Close"].isna().all():
            df[col_name] = data["Close"].squeeze().reindex(df.index)

    # 日米金利差: FRED API が利用可能なら正確な値、なければ US10Y のみ
    if HAS_FRED and config.get("FRED_API_KEY"):
        try:
            fred = Fred(api_key=config["FRED_API_KEY"])
            jp10y = fred.get_series("IRLTLT01JPM156N", observation_start=start)
            jp10y_daily = jp10y.resample("D").ffill().reindex(df.index, method="ffill")
            if "us10y" in df.columns:
                df["interest_rate_diff"] = df["us10y"] - jp10y_daily
            print("  金利差: FRED API 使用")
        except Exception as e:
            print(f"  FRED API error: {e}")

    print(f"  取得行数: {len(df)}")
    return df


# ===================================================================
# 2. 特徴量エンジニアリング (Section 3.1)
# ===================================================================
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """テクニカル指標を生成 (Section 3.1.2: time_varying_unknown)"""
    c, h, l = df["close"], df["high"], df["low"]

    # 対数リターン
    df["log_return"] = np.log(c / c.shift(1))

    # RSI(14)
    df["rsi_14"] = ta.momentum.RSIIndicator(c, window=14).rsi()

    # MACD
    macd_ind = ta.trend.MACD(c)
    df["macd"] = macd_ind.macd()
    df["macd_signal"] = macd_ind.macd_signal()
    df["macd_diff"] = macd_ind.macd_diff()

    # Bollinger Bands (20日, 2σ)
    bb = ta.volatility.BollingerBands(c, window=20, window_dev=2)
    df["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / c
    df["bb_pctb"] = bb.bollinger_pband()

    # ATR(14)
    df["atr_14"] = ta.volatility.AverageTrueRange(h, l, c, window=14).average_true_range()

    # 移動平均乖離率
    for w in [5, 20, 50, 200]:
        sma = c.rolling(w).mean()
        df[f"ma_dist_{w}"] = (c - sma) / sma

    # EMA乖離率
    for s in [12, 26]:
        ema = c.ewm(span=s, adjust=False).mean()
        df[f"ema_dist_{s}"] = (c - ema) / ema

    return df


def _compute_event_distances(
    idx: pd.DatetimeIndex, event_dates: list[str], max_distance: float = 45.0
) -> np.ndarray:
    """各日付からイベント日への最小絶対距離（日数）を計算 (searchsorted で O(n log m))

    max_distance でキャップすることで、イベントリスト外の期間でも
    正規化に悪影響を与えない安全な上限値を返す。
    """
    events = np.sort(pd.to_datetime(event_dates).values)
    idx_arr = idx.values
    if len(events) == 0:
        return np.full(len(idx), max_distance)

    pos = np.searchsorted(events, idx_arr)
    one_day = np.timedelta64(1, "D")

    # 右隣イベントへの距離
    right_idx = np.minimum(pos, len(events) - 1)
    dist_right = np.abs((events[right_idx] - idx_arr) / one_day).astype(np.float64)
    dist_right = np.where(pos < len(events), dist_right, max_distance)

    # 左隣イベントへの距離
    left_idx = np.maximum(pos - 1, 0)
    dist_left = np.abs((idx_arr - events[left_idx]) / one_day).astype(np.float64)
    dist_left = np.where(pos > 0, dist_left, max_distance)

    return np.clip(np.minimum(dist_right, dist_left), 0, max_distance)


def _nfp_dates(start_year: int, end_year: int) -> list[str]:
    """米国雇用統計（NFP）発表日リスト -BLS基準: 12日を含む週の翌金曜日

    実務上は「月初が金曜日なら第2金曜日、それ以外は第1金曜日」に近似。
    """
    from datetime import date, timedelta

    dates: list[str] = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            d = date(year, month, 1)
            days_until_friday = (4 - d.weekday()) % 7
            first_friday = d + timedelta(days=days_until_friday)
            # NFP は月初日そのものには発表されない (参照週が未完了)
            if first_friday.day == 1:
                first_friday += timedelta(days=7)
            dates.append(first_friday.isoformat())
    return dates


def add_event_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    """FOMC / BOJ / NFP イベント距離特徴量 (Section 3.1.1: time_varying_known)"""
    idx = df.index

    df["fomc_distance"] = _compute_event_distances(idx, FOMC_DATES)
    df["boj_distance"] = _compute_event_distances(idx, BOJ_DATES)

    nfp = _nfp_dates(idx[0].year, idx[-1].year + 1)
    df["nfp_distance"] = _compute_event_distances(idx, nfp)

    print(f"  イベント距離: FOMC({df['fomc_distance'].mean():.1f}d avg), "
          f"BOJ({df['boj_distance'].mean():.1f}d avg), "
          f"NFP({df['nfp_distance'].mean():.1f}d avg)")
    return df


def add_holiday_flags(df: pd.DataFrame) -> pd.DataFrame:
    """日米祝日フラグ (Section 3.1.1: time_varying_known)"""
    if HAS_HOLIDAYS:
        years = range(df.index[0].year, df.index[-1].year + 2)
        us_hol = _holidays_lib.UnitedStates(years=years)
        jp_hol = _holidays_lib.Japan(years=years)
        us_dates = pd.to_datetime(list(us_hol.keys()))
        jp_dates = pd.to_datetime(list(jp_hol.keys()))
        df["holiday_us"] = df.index.isin(us_dates).astype(float)
        df["holiday_jp"] = df.index.isin(jp_dates).astype(float)
        n_us = int(df["holiday_us"].sum())
        n_jp = int(df["holiday_jp"].sum())
        print(f"  祝日フラグ: US={n_us}日, JP={n_jp}日")
    else:
        print("  holidays ライブラリ未インストール -フォールバック (全0)")
        print("  インストール: uv add holidays")
        df["holiday_us"] = 0.0
        df["holiday_jp"] = 0.0
    return df


def add_news_sentiment_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """ニュースセンチメント代理変数 (Section 3.1.2: time_varying_unknown)

    本来は FinBERT / NewsAPI ベースだが、代替として VIX 乖離率を
    恐怖・楽観スコアに変換。将来の NLP モデル統合時に置き換え予定。
    """
    if "vix" in df.columns and df["vix"].notna().sum() > 20:
        vix_ma = df["vix"].rolling(20, min_periods=1).mean()
        df["sentiment_proxy"] = -(df["vix"] - vix_ma) / vix_ma.clip(lower=1e-6)
        df["sentiment_proxy"] = df["sentiment_proxy"].clip(-1, 1)
        print(f"  センチメント代理: VIXベース (mean={df['sentiment_proxy'].mean():.3f})")
    else:
        df["sentiment_proxy"] = 0.0
        print("  センチメント代理: VIX 欠損のためゼロ埋め")
    return df


def add_market_regime(df: pd.DataFrame) -> pd.DataFrame:
    """市場レジームラベル (Section 3.1.3)

    ADX(14) + 正規化 ATR でトレンド / レンジ / 高ボラティリティを分類。
    time_varying_unknown_categorical として TFT に入力。
    """
    c, h, l = df["close"], df["high"], df["low"]

    adx_val = ta.trend.ADXIndicator(h, l, c, window=14).adx()

    if "atr_14" in df.columns:
        atr_norm = df["atr_14"] / c
    else:
        atr_norm = ta.volatility.AverageTrueRange(h, l, c, window=14).average_true_range() / c

    vol_threshold = atr_norm.rolling(60, min_periods=1).quantile(0.75)

    # ADX ウォームアップ期間は NaN → adx_valid マスクで安全に処理
    adx_valid = adx_val.notna()
    regime = pd.Series("range", index=df.index)
    regime[adx_valid & (adx_val > 25)] = "trend"
    regime[adx_valid & (atr_norm > vol_threshold) & (adx_val <= 25)] = "high_vol"

    df["market_regime"] = regime.astype(str)

    counts = df["market_regime"].value_counts()
    print(f"  市場レジーム: {dict(counts)}")
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """カレンダー特徴量を生成 (Section 3.1.1: time_varying_known)"""
    idx = df.index

    # カテゴリ変数
    df["day_of_week"] = idx.dayofweek.astype(str)
    df["month"] = idx.month.astype(str)
    df["quarter"] = idx.quarter.astype(str)

    # 周期エンコーディング (known reals)
    df["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
    df["month_sin"] = np.sin(2 * np.pi * idx.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * idx.month / 12)
    df["doy_sin"] = np.sin(2 * np.pi * idx.dayofyear / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * idx.dayofyear / 365.25)

    # フラグ
    df["is_month_start"] = idx.is_month_start.astype(float)
    df["is_month_end"] = idx.is_month_end.astype(float)
    df["is_quarter_end"] = idx.is_quarter_end.astype(float)

    return df


# ===================================================================
# 3. 前処理 (Section 3.2)
# ===================================================================
def preprocess(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """データ前処理: 欠損値補完 → 外れ値クリッピング → NaN除去"""
    cfg = config if config is not None else CONFIG
    df = df.ffill()

    # 対数リターンの外れ値を ±4σ でクリッピング (訓練データのみから統計量を算出)
    if "log_return" in df.columns:
        train_end = int(len(df) * cfg["TRAIN_RATIO"])
        train_lr = df["log_return"].iloc[:train_end]
        mu, sigma = train_lr.mean(), train_lr.std()
        df["log_return"] = df["log_return"].clip(mu - 4 * sigma, mu + 4 * sigma)

    df = df.dropna()
    # 週末データを除外 (yfinance 結合時に紛れ込む可能性)
    df = df[df.index.dayofweek < 5]
    df = df.copy()

    # TimeSeriesDataSet 用の必須カラム
    df["time_idx"] = np.arange(len(df))
    df["group_id"] = "USDJPY"

    print(f"前処理後: {len(df)} rows ({df.index[0].date()} ~ {df.index[-1].date()})")
    return df


def prepare_data(config: dict) -> pd.DataFrame:
    """データ取得 → 特徴量エンジニアリング → 前処理の一括パイプライン"""
    df = fetch_market_data(config)
    df = add_technical_indicators(df)
    df = add_calendar_features(df)
    df = add_event_distance_features(df)
    df = add_holiday_flags(df)
    df = add_news_sentiment_proxy(df)
    df = add_market_regime(df)
    return preprocess(df, config)


# ===================================================================
# 4. TimeSeriesDataSet 作成
# ===================================================================
KNOWN_CATEGORICALS = ["day_of_week", "month", "quarter"]
KNOWN_REALS = [
    "time_idx",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "doy_sin",
    "doy_cos",
    "is_month_start",
    "is_month_end",
    "is_quarter_end",
    # イベント距離 (Section 3.1.1)
    "fomc_distance",
    "boj_distance",
    "nfp_distance",
    # 祝日フラグ (Section 3.1.1)
    "holiday_us",
    "holiday_jp",
]
UNKNOWN_REALS_BASE = [
    # R7-01: close除外（target=log_returnとの冗長性排除）
    "rsi_14",
    # R2-10: macd, macd_signal, bb_pctb, ema_dist_12, ema_dist_26 削除（冗長特徴量除去）
    "macd_diff",
    "bb_width",
    "atr_14",
    "ma_dist_5",
    "ma_dist_20",
    "ma_dist_50",
    "ma_dist_200",
]
OPTIONAL_UNKNOWN = ["vix", "gold", "us10y", "interest_rate_diff", "sentiment_proxy"]
UNKNOWN_CATEGORICALS = ["market_regime"]


def _build_unknown_reals(
    df: pd.DataFrame, config: dict | None = None, *, force_rebuild: bool = False
) -> list[str]:
    """利用可能な unknown reals を動的に構築し、確定した特徴量リストを返す。

    スキーマ安定性のため、欠損の多いオプション特徴量はNaN補完して含めるか除外する。
    一度確定したリストは artifacts/feature_schema.json に保存し、
    以降のセッションで同じスキーマを再利用する。

    オプション特徴量の欠損判定は訓練データのみで行う（テストデータリーク防止）。
    """
    _SCHEMA_VERSION = 3
    schema_path = ARTIFACT_DIR / "feature_schema.json"

    if not force_rebuild and schema_path.exists():
        with open(schema_path) as f:
            saved = json.load(f)
        if isinstance(saved, dict) and saved.get("version") == _SCHEMA_VERSION:
            cols = saved["columns"]
            if all(c in df.columns for c in cols):
                return cols

    # 訓練データのみで欠損判定（テストデータリーク防止）
    cfg = config if config is not None else CONFIG
    train_end = int(len(df) * cfg["TRAIN_RATIO"])
    df_train = df.iloc[:train_end]

    cols = list(UNKNOWN_REALS_BASE)
    for c in OPTIONAL_UNKNOWN:
        if c in df.columns and df_train[c].notna().sum() > len(df_train) * 0.5:
            cols.append(c)

    # 確定したスキーマを保存
    with open(schema_path, "w") as f:
        json.dump({"version": _SCHEMA_VERSION, "columns": cols}, f)

    return cols


def create_datasets(
    df: pd.DataFrame, config: dict
) -> tuple[TimeSeriesDataSet, TimeSeriesDataSet, TimeSeriesDataSet]:
    """Train / Validation / Test の TimeSeriesDataSet を作成"""
    n = len(df)
    train_end = int(n * config["TRAIN_RATIO"])
    val_end = int(n * (config["TRAIN_RATIO"] + config["VAL_RATIO"]))

    train_cutoff = int(df.iloc[train_end - 1]["time_idx"])
    val_cutoff = int(df.iloc[val_end - 1]["time_idx"])

    unknown_reals = _build_unknown_reals(df)

    # market_regime が存在する場合のみ unknown categorical に含める
    unknown_cats = [c for c in UNKNOWN_CATEGORICALS if c in df.columns]

    training = TimeSeriesDataSet(
        df[df["time_idx"] <= train_cutoff],
        time_idx="time_idx",
        target="log_return",  # R6-01: close→log_return
        group_ids=["group_id"],
        max_encoder_length=config["MAX_ENCODER_LENGTH"],
        min_encoder_length=3,  # R8-11: 5→3
        max_prediction_length=config["MAX_PREDICTION_LENGTH"],
        static_categoricals=["group_id"],
        time_varying_known_categoricals=KNOWN_CATEGORICALS,
        time_varying_known_reals=KNOWN_REALS,
        time_varying_unknown_categoricals=unknown_cats,
        time_varying_unknown_reals=unknown_reals,
        target_normalizer=EncoderNormalizer(),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    # Validation: encoder は訓練期間にも跨がるため、train_cutoff 以前のデータも含める
    validation = TimeSeriesDataSet.from_dataset(
        training,
        df[df["time_idx"] <= val_cutoff],
        min_prediction_idx=train_cutoff + 1,
        stop_randomization=True,
    )

    # Test
    test = TimeSeriesDataSet.from_dataset(
        training,
        df,
        min_prediction_idx=val_cutoff + 1,
        stop_randomization=True,
    )

    print(f"特徴量 -Known cat: {len(KNOWN_CATEGORICALS)}, Known real: {len(KNOWN_REALS)}, "
          f"Unknown cat: {len(unknown_cats)}, Unknown real: {len(unknown_reals)}")
    print(f"データセット -Train: {len(training)}, Val: {len(validation)}, Test: {len(test)}")

    return training, validation, test


# ===================================================================
# 5. 学習 (Section 4)
# ===================================================================
def train_tft(
    training: TimeSeriesDataSet,
    validation: TimeSeriesDataSet,
    config: dict,
    max_epochs: int | None = None,
    fold_id: str | None = None,
) -> tuple[TemporalFusionTransformer, pl.Trainer]:
    """TFT モデル学習 -QuantileLoss + AdamW + EarlyStopping + AMP"""
    train_dl = training.to_dataloader(
        train=True, batch_size=config["BATCH_SIZE"], num_workers=0,
        pin_memory=PIN_MEMORY,
    )
    val_dl = validation.to_dataloader(
        train=False, batch_size=config["BATCH_SIZE"], num_workers=0,
        pin_memory=PIN_MEMORY,
    )

    ckpt_subdir = f"checkpoints_{fold_id}" if fold_id else "checkpoints"
    ckpt_dir_path = ARTIFACT_DIR / ckpt_subdir
    ckpt_dir_path.mkdir(parents=True, exist_ok=True)
    for old_ckpt in ckpt_dir_path.glob("*.ckpt"):
        old_ckpt.unlink()
    ckpt_dir = str(ckpt_dir_path)
    early_stop = EarlyStopping(
        monitor="val_loss", patience=config["PATIENCE"], mode="min", verbose=True
    )
    ckpt = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="tft-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs or config["MAX_EPOCHS"],
        accelerator="auto",
        precision="bf16-mixed" if PIN_MEMORY else "32-true",
        gradient_clip_val=config["GRADIENT_CLIP_VAL"],
        accumulate_grad_batches=2,
        callbacks=[early_stop, LearningRateMonitor(), ckpt],
        enable_progress_bar=True,
    )

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=config["LEARNING_RATE"],
        hidden_size=config["HIDDEN_SIZE"],
        attention_head_size=config["ATTENTION_HEAD_SIZE"],
        dropout=config["DROPOUT"],
        hidden_continuous_size=config["HIDDEN_CONTINUOUS_SIZE"],
        loss=DirectionAwareQuantileLoss(
            quantiles=config["QUANTILES"],
            direction_weight=config.get("DIRECTION_LOSS_WEIGHT", 0.0),
        ),
        optimizer="adamw",
        weight_decay=5e-3,
        reduce_on_plateau_patience=4,
    )
    print(f"パラメータ数: {tft.size() / 1e3:.1f}k")

    trainer.fit(tft, train_dataloaders=train_dl, val_dataloaders=val_dl)

    best_path = ckpt.best_model_path
    if best_path:
        best_model = TemporalFusionTransformer.load_from_checkpoint(best_path)
    else:
        best_model = tft

    return best_model, trainer


# ===================================================================
# 6. 評価 (Section 5.1)
# ===================================================================
def evaluate(
    model: TemporalFusionTransformer,
    dataset: TimeSeriesDataSet,
    config: dict,
) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """モデル評価: MAE / RMSE / MAPE / 方向精度 / 分位点カバレッジ

    Returns: (metrics, preds, actuals, encoder_last)
    全て denormalized (原価格スケール)。
    Note: pytorch-forecasting の y[0] / encoder_target は既に原価格スケール。
    メトリクス計算は GPU 上で実行し、最終 .item() のみ CPU に転送。
    """
    dl = dataset.to_dataloader(
        train=False, batch_size=config["BATCH_SIZE"], num_workers=0,
        pin_memory=PIN_MEMORY,
    )

    # Predictions (denormalized): [N, pred_len, n_quantiles]
    # mode="quantiles" で全分位点を3D tensorとして取得 (GPU上)
    preds = model.predict(dl, mode="quantiles")

    # Actuals / encoder_last: y[0], encoder_target は既に原価格スケール
    # GPU に転送して保持
    actuals_list = []
    enc_last_list = []
    for x, y in dl:
        actuals_list.append(y[0].to(DEVICE, non_blocking=True))
        if "encoder_target" in x:
            enc_last_list.append(x["encoder_target"][:, -1].to(DEVICE, non_blocking=True))

    actuals = torch.cat(actuals_list)
    encoder_last = torch.cat(enc_last_list) if enc_last_list else None

    # preds も GPU に揃える
    preds = preds.to(DEVICE, non_blocking=True)
    n_q = preds.size(2)
    q_mid = n_q // 2  # median index (3 for 7 quantiles, 1 for 3)
    q50 = preds[:, :, q_mid]
    q_lo = preds[:, :, 0]   # lowest quantile
    q_hi = preds[:, :, -1]  # highest quantile

    # --- GPU 上でメトリクス計算 ---
    diff = q50 - actuals
    mae = diff.abs().mean().item()
    rmse = (diff ** 2).mean().sqrt().item()

    # --- 方向精度 (GPU) ---
    # R6-01: target=log_return: 符号で方向判定 (>0 = 上昇)
    # 1日先
    actual_dir_1d = actuals[:, 0] > 0
    pred_dir_1d = q50[:, 0] > 0
    direction_acc = (actual_dir_1d == pred_dir_1d).float().mean().item()
    # 5日先 (累積リターン: 1〜5日の合計で方向判定)
    if actuals.size(1) >= 5:
        actual_cum_5d = actuals[:, :5].sum(dim=1)
        pred_cum_5d = q50[:, :5].sum(dim=1)
        direction_acc_5d = ((actual_cum_5d > 0) == (pred_cum_5d > 0)).float().mean().item()
    else:
        direction_acc_5d = direction_acc

    # --- 分位点カバレッジ: 最外分位点間 (GPU) ---
    coverage = ((actuals >= q_lo) & (actuals <= q_hi)).float().mean().item()

    metrics: dict = {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "direction_accuracy": round(direction_acc, 4),
        "direction_accuracy_5d": round(direction_acc_5d, 4),
        "quantile_coverage_80": round(coverage, 4),
    }

    # ホライズン別 MAE (1d, 5d, 10d, 20d)
    for h_idx, h_name in [(0, "1d"), (4, "5d"), (9, "10d"), (19, "20d")]:
        if h_idx < preds.size(1):
            h_mae = (q50[:, h_idx] - actuals[:, h_idx]).abs().mean().item()
            metrics[f"mae_{h_name}"] = round(h_mae, 4)

    return metrics, preds, actuals, encoder_last


# ===================================================================
# 7. トレーディングバックテスト (Section 5.2)
# ===================================================================
def backtest_trading(
    preds: torch.Tensor,
    actuals: torch.Tensor,
    encoder_last: torch.Tensor | None,
    config: dict,
) -> dict:
    """簡易トレーディングバックテスト (GPU 上で計算)

    R6-01: target=log_return: 予測値の符号でシグナル生成。
    PnL は log_return * 想定ポジション。
    """
    if preds.size(0) < 2:
        return {"error": "insufficient_samples"}

    # GPU 上で全計算
    dev = preds.device
    q_mid = preds.size(2) // 2
    pred_1d = preds[:, 0, q_mid]   # 1日先 predicted log_return
    actual_1d = actuals[:, 0]      # 1日先 actual log_return

    cost = (config["SPREAD_PIPS"] + config["SLIPPAGE_PIPS"]) * config["PIP_SIZE"]

    # シグナル: predicted log_return の符号
    signals = torch.zeros_like(pred_1d)
    signals[pred_1d > 0] = 1.0   # Buy
    signals[pred_1d < 0] = -1.0  # Sell

    # PnL: signals * actual_log_return（log_return は近似的に price_change/price）
    # cost は price スケールなので log_return スケールに変換（÷150 程度、概算）
    cost_lr = cost / 150.0  # 概算の log_return スケールコスト
    pnl = signals * actual_1d - signals.abs() * cost_lr
    traded = signals != 0
    n_trades = int(traded.sum().item())

    if n_trades > 0:
        trade_pnl = pnl[traded]
        win_rate = float((trade_pnl > 0).sum().item() / n_trades)
        gross_profit = float(trade_pnl[trade_pnl > 0].sum().item())
        gross_loss = float(trade_pnl[trade_pnl < 0].abs().sum().item())
        profit_factor = gross_profit / max(gross_loss, 1e-10)
    else:
        win_rate, profit_factor = 0.0, 0.0

    cum_pnl = pnl.cumsum(0)
    running_max = cum_pnl.cummax(0).values
    max_dd = float((cum_pnl - running_max).min().item())

    # Sharpe ratio: 全日を含める (非取引日は0リターン)
    sharpe = float(
        pnl.mean().item() / max(pnl.std().item(), 1e-10) * (252 ** 0.5)
    )

    return {
        "total_pnl": round(float(cum_pnl[-1].item()), 2),
        "n_trades": n_trades,
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 4),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown": round(max_dd, 4),
    }


# ===================================================================
# 8. モデル解釈 -TFT の変数重要度 + Attention
# ===================================================================
def interpret(
    model: TemporalFusionTransformer,
    dataset: TimeSeriesDataSet,
    config: dict,
) -> dict | None:
    """TFT の解釈可能性: Variable Selection Network の重要度を取得"""
    dl = dataset.to_dataloader(
        train=False, batch_size=config["BATCH_SIZE"], num_workers=0
    )
    try:
        raw_preds = model.predict(dl, mode="raw", return_x=True)
        # predict(return_x=True) は Prediction namedtuple を返す
        # interpret_output は raw output dict を期待するため .output を渡す
        raw_output = raw_preds.output if hasattr(raw_preds, "output") else raw_preds
        interpretation = model.interpret_output(raw_output, reduction="sum")

        print("\n--- 変数重要度 ---")
        for section in ["encoder_variables", "decoder_variables", "static_variables"]:
            if section in interpretation:
                imp = interpretation[section]
                if hasattr(imp, "items"):
                    sorted_imp = sorted(imp.items(), key=lambda kv: kv[1], reverse=True)
                    print(f"\n{section}:")
                    for name, val in sorted_imp[:10]:
                        print(f"  {name}: {val:.4f}")

        return interpretation
    except Exception as e:
        print(f"  解釈失敗: {e}")
        return None


# ===================================================================
# 9. Walk-Forward バックテスト (Section 5.2)
# ===================================================================
def walk_forward_backtest(df: pd.DataFrame, config: dict) -> dict | None:
    """ウォークフォワードバックテスト

    初期学習: WF_INITIAL_TRAIN_DAYS 日
    テストウィンドウ: WF_TEST_WINDOW 日ずつスライド
    各 fold で fine-tuning (WF_FINETUNE_EPOCHS エポック)
    """
    initial_train = config["WF_INITIAL_TRAIN_DAYS"]
    test_window = config["WF_TEST_WINDOW"]
    n = len(df)

    if n < initial_train + test_window:
        print("データ不足: ウォークフォワード不可")
        return None

    unknown_reals = _build_unknown_reals(df)
    unknown_cats = [c for c in UNKNOWN_CATEGORICALS if c in df.columns]
    all_metrics: list[dict] = []
    cursor = initial_train
    fold = 0

    while cursor + test_window <= n:
        fold += 1
        full_slice = df.iloc[: cursor + test_window].copy()

        # time_idx を fold 毎にリセット
        full_slice["time_idx"] = np.arange(len(full_slice))

        # 訓練データからさらに val 分割 (最後10%を検証用、test_ds は未使用)
        val_size = max(int(cursor * 0.1), config["MAX_ENCODER_LENGTH"] + config["MAX_PREDICTION_LENGTH"])
        train_end_idx = cursor - val_size
        train_cutoff_wf = int(full_slice.iloc[train_end_idx - 1]["time_idx"])
        val_cutoff_wf = int(full_slice.iloc[cursor - 1]["time_idx"])
        test_start_wf = int(full_slice.iloc[cursor]["time_idx"])

        try:
            pred_len = min(config["MAX_PREDICTION_LENGTH"], test_window)
            training = TimeSeriesDataSet(
                full_slice[full_slice["time_idx"] <= train_cutoff_wf],
                time_idx="time_idx",
                target="log_return",  # R6-01
                group_ids=["group_id"],
                max_encoder_length=config["MAX_ENCODER_LENGTH"],
                min_encoder_length=3,  # create_datasets() と統一
                max_prediction_length=pred_len,
                static_categoricals=["group_id"],
                time_varying_known_categoricals=KNOWN_CATEGORICALS,
                time_varying_known_reals=KNOWN_REALS,
                time_varying_unknown_categoricals=unknown_cats,
                time_varying_unknown_reals=unknown_reals,
                target_normalizer=EncoderNormalizer(),
                add_relative_time_idx=True,
                add_target_scales=True,
                add_encoder_length=True,
            )

            # EarlyStopping 用の検証セット (訓練期間の最後10%)
            val_ds = TimeSeriesDataSet.from_dataset(
                training,
                full_slice[full_slice["time_idx"] <= val_cutoff_wf],
                min_prediction_idx=train_cutoff_wf + 1,
                stop_randomization=True,
            )

            # テストセット (fold の未来ウィンドウ) -評価のみに使用
            test_ds = TimeSeriesDataSet.from_dataset(
                training,
                full_slice,
                min_prediction_idx=test_start_wf,
                stop_randomization=True,
            )

            if len(test_ds) == 0 or len(val_ds) == 0:
                cursor += test_window
                continue

            # val_ds で EarlyStopping → test_ds は未接触
            model, _ = train_tft(
                training, val_ds, config, max_epochs=config["WF_FINETUNE_EPOCHS"]
            )
            metrics, _, _, _ = evaluate(model, test_ds, config)
            metrics["fold"] = fold
            metrics["train_end"] = str(df.index[cursor - 1].date())
            all_metrics.append(metrics)

            print(
                f"  Fold {fold}: MAE={metrics['mae']:.4f}, "
                f"Dir={metrics['direction_accuracy']:.4f}"
            )

        except Exception as e:
            print(f"  Fold {fold} failed: {e}")

        cursor += test_window
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not all_metrics:
        return None

    numeric_keys = [
        k
        for k in all_metrics[0]
        if isinstance(all_metrics[0].get(k), (int, float))
    ]
    avg = {k: float(np.mean([m[k] for m in all_metrics])) for k in numeric_keys}

    return {"folds": all_metrics, "average": avg}


# ===================================================================
# 10. Optuna ハイパラチューニング (Section 4.3)
# ===================================================================
def optuna_optimize(df: pd.DataFrame, config: dict) -> dict:
    """Optuna TPE Sampler でハイパーパラメータ最適化"""
    print("\n=== Optuna Hyperparameter Search ===")

    def objective(trial: optuna.Trial) -> float:
        trial_config = config.copy()
        trial_config["HIDDEN_SIZE"] = trial.suggest_categorical(
            "hidden_size", [64, 128, 160, 256]
        )
        trial_config["ATTENTION_HEAD_SIZE"] = trial.suggest_categorical(
            "attention_head_size", [1, 2, 4]
        )
        trial_config["DROPOUT"] = trial.suggest_float("dropout", 0.05, 0.3)
        trial_config["HIDDEN_CONTINUOUS_SIZE"] = trial.suggest_categorical(
            "hidden_continuous_size", [4, 8, 16]
        )
        trial_config["LEARNING_RATE"] = trial.suggest_float(
            "learning_rate", 1e-4, 3e-3, log=True
        )

        try:
            training, validation, _ = create_datasets(df, trial_config)
            model, _ = train_tft(training, validation, trial_config, max_epochs=30)
            m, _, _, _ = evaluate(model, validation, trial_config)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"  Trial {trial.number}: dir_5d={m['direction_accuracy_5d']:.4f}")
            return m["direction_accuracy_5d"]

        except Exception as e:
            print(f"  Trial {trial.number} failed: {e}")
            return 0.0

    study = optuna.create_study(
        direction="maximize",  # 方向精度を最大化
        sampler=optuna.samplers.TPESampler(seed=config["RANDOM_SEED"]),
    )
    study.optimize(
        objective,
        n_trials=config["N_TRIALS"],
        timeout=config["OPTUNA_TIMEOUT"],
        show_progress_bar=True,
    )

    print(f"\nBest direction_5d: {study.best_value:.4f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    return study.best_params


# ===================================================================
# 11. メインパイプライン
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description="TFT USD/JPY Prediction System")
    parser.add_argument("--optuna", action="store_true", help="Optuna チューニング")
    parser.add_argument(
        "--walkforward", action="store_true", help="Walk-Forward バックテスト"
    )
    parser.add_argument(
        "--deploy", action="store_true",
        help="学習後にベストモデルを dashboard/model/ にデプロイ",
    )
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    pl.seed_everything(CONFIG["RANDOM_SEED"])

    print(f"Device: {DEVICE}")
    print(f"Artifacts: {ARTIFACT_DIR}\n")

    # ── 1-3. データ収集・特徴量・前処理 ──
    df = prepare_data(CONFIG)

    # ── 4. Optuna (optional) ──
    if args.optuna:
        best_params = optuna_optimize(df, CONFIG)
        CONFIG["HIDDEN_SIZE"] = best_params.get("hidden_size", CONFIG["HIDDEN_SIZE"])
        CONFIG["ATTENTION_HEAD_SIZE"] = best_params.get(
            "attention_head_size", CONFIG["ATTENTION_HEAD_SIZE"]
        )
        CONFIG["DROPOUT"] = best_params.get("dropout", CONFIG["DROPOUT"])
        CONFIG["HIDDEN_CONTINUOUS_SIZE"] = best_params.get(
            "hidden_continuous_size", CONFIG["HIDDEN_CONTINUOUS_SIZE"]
        )
        CONFIG["LEARNING_RATE"] = best_params.get(
            "learning_rate", CONFIG["LEARNING_RATE"]
        )

    # ── 5. データセット作成 ──
    print("\n=== 4. データセット作成 ===")
    training, validation, test = create_datasets(df, CONFIG)

    # ── 6. マルチseedアンサンブル学習 ──
    n_seeds = CONFIG.get("ENSEMBLE_SEEDS", 5)
    base_seed = CONFIG["RANDOM_SEED"]
    seeds = [base_seed + i * 137 for i in range(n_seeds)]

    print(f"\n=== 5. TFT アンサンブル学習 ({n_seeds} seeds) ===")
    sum_preds = None
    best_model = None
    best_val_dir = -1.0

    for i, seed in enumerate(seeds):
        pl.seed_everything(seed)
        print(f"\n--- Seed {seed} ({i+1}/{n_seeds}) ---")
        model, trainer = train_tft(training, validation, CONFIG)
        m_val, _, _, _ = evaluate(model, validation, CONFIG)
        m_test, preds_i, actuals, encoder_last = evaluate(model, test, CONFIG)
        if sum_preds is None:
            sum_preds = preds_i.clone()
        else:
            sum_preds += preds_i
        del preds_i
        print(f"  val_dir={m_val['direction_accuracy']:.4f}  test_dir={m_test['direction_accuracy']:.4f}  test_5d={m_test['direction_accuracy_5d']:.4f}")

        if m_val["direction_accuracy"] > best_val_dir:
            best_val_dir = m_val["direction_accuracy"]
            best_model = model
        else:
            del model
        if PIN_MEMORY:
            torch.cuda.empty_cache()

    ens_preds = sum_preds / n_seeds
    q_mid = ens_preds.size(2) // 2
    q50_ens = ens_preds[:, :, q_mid].to(DEVICE)
    actuals_dev = actuals.to(DEVICE)

    # R6-01: target=log_return — 符号で方向判定
    ens_dir_1d = ((actuals_dev[:, 0] > 0) == (q50_ens[:, 0] > 0)).float().mean().item()
    # 5日先: 累積リターン (1〜5日の合計) で方向判定
    if actuals_dev.size(1) >= 5:
        ens_cum_5d_act = actuals_dev[:, :5].sum(dim=1)
        ens_cum_5d_pred = q50_ens[:, :5].sum(dim=1)
        ens_dir_5d = ((ens_cum_5d_act > 0) == (ens_cum_5d_pred > 0)).float().mean().item()
    else:
        ens_dir_5d = ens_dir_1d

    # ── 7. テスト評価 (アンサンブル) ──
    print(f"\n=== 6. アンサンブル評価 ({n_seeds} seeds) ===")
    # 個別best の評価
    metrics, preds, actuals, encoder_last = evaluate(best_model, test, CONFIG)
    # アンサンブル方向精度で上書き
    metrics["ensemble_direction_1d"] = round(ens_dir_1d, 4)
    metrics["ensemble_direction_5d"] = round(ens_dir_5d, 4)
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # ── 8. トレーディングバックテスト ──
    print("\n=== 7. トレーディングバックテスト ===")
    trade_metrics = backtest_trading(ens_preds.to(DEVICE), actuals_dev, encoder_last, CONFIG)
    for k, v in trade_metrics.items():
        print(f"  {k}: {v}")

    # ── 9. モデル解釈 (best single model) ──
    print("\n=== 8. モデル解釈 ===")
    model = best_model
    interpret(model, test, CONFIG)

    # ── 10. Walk-Forward (optional) ──
    wf_results = None
    if args.walkforward:
        print("\n=== 9. Walk-Forward バックテスト ===")
        wf_results = walk_forward_backtest(df, CONFIG)
        if wf_results:
            print("\nWalk-Forward 平均:")
            for k, v in wf_results["average"].items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")

    # ── 11. アーティファクト保存 ──
    print("\n=== アーティファクト保存 ===")
    all_metrics = {**metrics, **{f"trade_{k}": v for k, v in trade_metrics.items()}}

    with open(ARTIFACT_DIR / "tft_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    # ダッシュボードのキャッシュを無効化（新モデルの結果で再計算させる）
    dashboard_cache = BASE_DIR / "dashboard" / "backtest_cache.json"
    if dashboard_cache.exists():
        dashboard_cache.unlink()
        print(f"  Cleared: {dashboard_cache}")

    _SECRET_KEYS = {"FRED_API_KEY"}
    serializable_config = {}
    for k, v in CONFIG.items():
        if k in _SECRET_KEYS:
            continue
        if isinstance(v, (str, int, float, bool, list)) or v is None:
            serializable_config[k] = v
    with open(ARTIFACT_DIR / "tft_config.json", "w") as f:
        json.dump(serializable_config, f, indent=2, default=str)

    if wf_results:
        with open(ARTIFACT_DIR / "tft_walkforward.json", "w") as f:
            json.dump(wf_results, f, indent=2, default=str)

    print(f"Artifacts: {ARTIFACT_DIR}")
    print(f"\n{'='*50}")
    print(f"MAE: {metrics['mae']:.4f}  |  Direction: {metrics['direction_accuracy']:.4f}")
    print(f"Dir_5d: {metrics.get('direction_accuracy_5d', 'N/A')}  |  Ens_dir_5d: {metrics.get('ensemble_direction_5d', 'N/A')}")
    print(f"Sharpe: {trade_metrics.get('sharpe_ratio', 'N/A')}  |  Trades: {trade_metrics.get('n_trades', 0)}")
    print(f"PF: {trade_metrics.get('profit_factor', 'N/A')}  |  WinRate: {trade_metrics.get('win_rate', 'N/A')}")
    print(f"{'='*50}")

    # ── 最適化ログ追記 (ralph-loop 用) ──
    opt_log_path = ARTIFACT_DIR / "optimization_log.json"
    opt_log: list = []
    if opt_log_path.exists():
        try:
            with open(opt_log_path) as f:
                opt_log = json.load(f)
        except (json.JSONDecodeError, ValueError):
            opt_log = []

    from datetime import datetime
    opt_entry = {
        "iteration": len(opt_log) + 1,
        "timestamp": datetime.now().isoformat(),
        "metrics": all_metrics,
        "config_snapshot": {
            k: v for k, v in CONFIG.items()
            if k not in {"FRED_API_KEY"} and (isinstance(v, (str, int, float, bool, list)) or v is None)
        },
    }
    opt_log.append(opt_entry)
    with open(opt_log_path, "w") as f:
        json.dump(opt_log, f, indent=2, ensure_ascii=False, default=str)
    print(f"Optimization log: iteration {opt_entry['iteration']} saved")

    # ── ダッシュボードへのモデルデプロイ ──
    if args.deploy:
        import shutil
        deploy_dir = BASE_DIR / "dashboard" / "model"
        deploy_dir.mkdir(parents=True, exist_ok=True)
        # 既存モデルをクリア
        for old in deploy_dir.glob("*.ckpt"):
            old.unlink()
        # ベストモデルのチェックポイントをコピー
        ckpt_dir = ARTIFACT_DIR / "checkpoints"
        ckpts = list(ckpt_dir.glob("*.ckpt"))
        if ckpts:
            best_ckpt = min(ckpts, key=lambda p: float(
                p.stem.split("val_loss=")[1].split("-v")[0]
            ) if "val_loss=" in p.stem else float("inf"))
            dest = deploy_dir / best_ckpt.name
            shutil.copy2(best_ckpt, dest)
            # メトリクスも一緒にコピー
            for fname in ["tft_metrics.json", "tft_config.json", "feature_schema.json"]:
                src = ARTIFACT_DIR / fname
                if src.exists():
                    shutil.copy2(src, deploy_dir / fname)
            print(f"Deployed to dashboard: {dest.name}")
        else:
            print("Warning: No checkpoint found for deploy")

    return all_metrics


if __name__ == "__main__":
    main()
