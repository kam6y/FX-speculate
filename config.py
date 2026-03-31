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
HIDDEN_SIZE = 32
ATTENTION_HEAD_SIZE = 4
DROPOUT = 0.15
HIDDEN_CONTINUOUS_SIZE = 16
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]
OUTPUT_SIZE = len(QUANTILES)

# --- Loss ---
DIRECTION_WEIGHT = 0.0  # Stage 1: 純粋 QuantileLoss
SMOOTHING_TEMPERATURE = 2.0
DEAD_ZONE = 1e-4

# --- Training ---
MAX_EPOCHS = 100
EARLY_STOP_PATIENCE = 20
LEARNING_RATE = 1e-4
BATCH_SIZE = 32

# --- Stage 2: 方向ファインチューニング ---
FINETUNE_DIRECTION_WEIGHT = 0.5
FINETUNE_LEARNING_RATE = 2e-5
FINETUNE_MAX_EPOCHS = 10
FINETUNE_HORIZON_WEIGHTS = [2.0, 1.5, 1.0, 1.0, 1.0]
assert len(FINETUNE_HORIZON_WEIGHTS) == PREDICTION_LENGTH, (
    f"FINETUNE_HORIZON_WEIGHTS length {len(FINETUNE_HORIZON_WEIGHTS)} "
    f"must match PREDICTION_LENGTH {PREDICTION_LENGTH}"
)

# --- Inference ---
QUANTILE_SIGNAL_WEIGHTS = [0.05, 0.15, 0.30, 0.25, 0.25]  # 上位テール重みで方向シグナル強化
assert len(QUANTILE_SIGNAL_WEIGHTS) == len(QUANTILES), (
    f"QUANTILE_SIGNAL_WEIGHTS length {len(QUANTILE_SIGNAL_WEIGHTS)} "
    f"must match QUANTILES length {len(QUANTILES)}"
)
TOP_K_CHECKPOINTS = 5
RANDOM_SEED = 42
DATA_SPLIT_RATIOS = {
    "train": 0.65,
    "val": 0.15,
    "threshold_tune": 0.05,
    "test": 0.15,
}
