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


def migrate_db(db_path) -> None:
    """predictions テーブルに actual 系カラムを idempotent に追加する。"""
    with sqlite3.connect(str(db_path)) as conn:
        existing = {
            row[1]
            for row in conn.execute("PRAGMA table_info(predictions)").fetchall()
        }
        for col, col_type in [
            ("actual_return", "REAL"),
            ("actual_direction", "TEXT"),
            ("is_correct", "INTEGER"),
        ]:
            if col not in existing:
                conn.execute(
                    f"ALTER TABLE predictions ADD COLUMN {col} {col_type}"
                )


def backfill_actuals(db_path, price_series: pd.Series) -> None:
    """過去予測に実績リターンを書き戻す。

    Args:
        db_path: predictions.db のパス
        price_series: 日付インデックスの USD/JPY 終値 Series
    """
    today_str = str(date.today())

    with sqlite3.connect(str(db_path)) as conn:
        rows = conn.execute(
            "SELECT rowid, prediction_date, target_date, direction "
            "FROM predictions "
            "WHERE actual_return IS NULL AND target_date <= ?",
            (today_str,),
        ).fetchall()

        # price_series のインデックスを文字列キーでも引けるよう正規化
        price_index = {
            str(dt.date()) if hasattr(dt, "date") else str(dt): price
            for dt, price in price_series.items()
        }

        for rowid, pred_date, tgt_date, direction in rows:
            close_pred = price_index.get(pred_date)
            close_tgt = price_index.get(tgt_date)
            if close_pred is None or close_tgt is None:
                continue

            actual_return = float(np.log(close_tgt / close_pred))
            actual_direction = "UP" if actual_return > 0 else "DOWN"
            is_correct = 1 if direction == actual_direction else 0

            conn.execute(
                "UPDATE predictions "
                "SET actual_return=?, actual_direction=?, is_correct=? "
                "WHERE rowid=?",
                (actual_return, actual_direction, is_correct, rowid),
            )


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
    # DB マイグレーション (actual 系カラム追加)
    if PREDICTIONS_DB.exists():
        migrate_db(PREDICTIONS_DB)

    print("=== 最新データ取得 ===")
    raw = fetch_all_data(use_cache=False)

    # 過去予測に実績を書き戻す
    if PREDICTIONS_DB.exists() and "usdjpy_close" in raw.columns:
        backfill_actuals(PREDICTIONS_DB, raw["usdjpy_close"])

    features = build_features(raw)
    prepped = prepare_data(features)

    # 学習用データセットの構築（normalizer 再利用のため）
    train_df, val_df, _, _ = split_data(prepped)
    training, _ = create_datasets(train_df, val_df)

    # encoder_data: 最新 ENCODER_LENGTH(=90) 営業日
    if len(prepped) < ENCODER_LENGTH:
        raise ValueError(f"データが不足しています: {len(prepped)} < {ENCODER_LENGTH}")
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
    # time_idx を 0 から振り直す（training の time_idx 範囲と乖離しないよう）
    combined["time_idx"] = range(len(combined))

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
        # direction_signal (全分位点の加重平均) で方向判定
        dir_signal = float(preds["direction_signal"][0, h])
        direction = "UP" if dir_signal > threshold else "DOWN"

        results.append({
            "prediction_date": str(date.today()),
            "target_date": str(future_dates[h].date()),
            "horizon": h + 1,
            "median": median_val,
            "direction_signal": dir_signal,
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
    with sqlite3.connect(str(PREDICTIONS_DB)) as conn:
        results_df.to_sql("predictions", conn, if_exists="append", index=False)
    print(f"\n  Saved to {PREDICTIONS_DB}")


if __name__ == "__main__":
    predict_daily()
