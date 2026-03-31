"""pytorch-forecasting 用 TimeSeriesDataSet 構築。"""

import pandas as pd
import numpy as np
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

from config import (
    ENCODER_LENGTH,
    PREDICTION_LENGTH,
    DATA_SPLIT_RATIOS,
)

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
    """TimeSeriesDataSet 用にデータを整形する。"""
    result = df.copy()
    result["time_idx"] = range(len(result))
    result["group_id"] = "usdjpy"
    for col in TIME_VARYING_KNOWN_CATEGORICALS:
        if col in result.columns:
            result[col] = result[col].astype(str)
    return result


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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


def create_datasets(train_df: pd.DataFrame, val_df: pd.DataFrame) -> tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
    """学習用・バリデーション用の TimeSeriesDataSet を作成する。"""
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
