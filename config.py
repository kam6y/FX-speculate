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
