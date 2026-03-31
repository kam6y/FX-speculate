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
    ATTENTION_HEAD_SIZE,
    BATCH_SIZE,
    HIDDEN_CONTINUOUS_SIZE,
    OPTUNA_DB,
    ENCODER_LENGTH,
    PREDICTION_LENGTH,
    OUTPUT_SIZE,
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
        "best_val_loss": float(trainer.checkpoint_callback.best_model_score) if trainer.checkpoint_callback.best_model_score is not None else float("nan"),
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
            attention_head_size=ATTENTION_HEAD_SIZE,
            dropout=dropout,
            hidden_continuous_size=HIDDEN_CONTINUOUS_SIZE,
            output_size=OUTPUT_SIZE,
            loss=DirectionAwareQuantileLoss(direction_weight=direction_weight),
            reduce_on_plateau_patience=4,
        )

        trainer = build_trainer(max_epochs=30)
        trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

        score = trainer.checkpoint_callback.best_model_score
        return float(score) if score is not None else float("inf")

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
