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
    QUANTILE_SIGNAL_WEIGHTS,
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
                try:
                    return float(part.split("=")[1])
                except ValueError:
                    continue
        return float("inf")

    ckpts.sort(key=parse_val_loss)
    return ckpts[:top_k]


def ensemble_predict(
    models: list[TemporalFusionTransformer],
    dataloader,
) -> dict:
    """top-k モデルのアンサンブル予測。

    - median: q50 平均
    - q10: 各モデルの min
    - q90: 各モデルの max
    - direction_signal: 全分位点の加重平均（分布の歪み情報を活用）
    """
    all_preds = []
    for model in models:
        preds = model.predict(dataloader, mode="quantiles", return_x=False)
        all_preds.append(preds)

    stacked = torch.stack(all_preds)  # (n_models, batch, horizon, quantiles)
    q_mid = QUANTILES.index(0.5)

    quantile_weights = torch.tensor(QUANTILE_SIGNAL_WEIGHTS, device=stacked.device)
    model_mean = stacked.mean(dim=0)  # (batch, horizon, quantiles)
    direction_signal = (model_mean * quantile_weights).sum(dim=-1)

    q10_idx = QUANTILES.index(0.1)
    q90_idx = QUANTILES.index(0.9)

    result = {
        "median": model_mean[..., q_mid],
        "q10": stacked[:, :, :, q10_idx].min(dim=0).values,
        "q90": stacked[:, :, :, q90_idx].max(dim=0).values,
        "direction_signal": direction_signal,
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
    """方向精度を最大化しつつ ratio_gap を抑える閾値を探索する。

    score = accuracy - 0.5 * ratio_gap のバランスで最適化。
    """
    actual_up = (actuals > 0).astype(bool)
    actual_up_ratio = actual_up.mean()

    if predictions.min() == predictions.max():
        return 0.0

    thresholds_grid = np.linspace(predictions.min(), predictions.max(), 1000)
    pred_up_matrix = predictions[None, :] > thresholds_grid[:, None]  # (1000, N)
    accuracy_vec = (pred_up_matrix == actual_up[None, :]).mean(axis=1)
    ratio_gap_vec = np.abs(actual_up_ratio - pred_up_matrix.mean(axis=1))
    scores = accuracy_vec - 0.5 * ratio_gap_vec

    return float(thresholds_grid[np.argmax(scores)])


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
    # 全実績値を一度だけ収集
    tune_actual_all = torch.stack([
        y for (_, (y, _)) in tune_loader.dataset
    ])
    for h in range(PREDICTION_LENGTH):
        preds_h = tune_preds["direction_signal"][:, h].numpy()
        actuals_h = tune_actual_all[:len(preds_h), h].numpy()
        thresholds[f"horizon_{h+1}"] = float(find_optimal_threshold(preds_h, actuals_h))

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

    # 全実績値を一度だけ収集（ホライズンループの外）
    actual_all = torch.stack([
        y for (_, (y, _)) in test_loader.dataset
    ])

    for h in range(PREDICTION_LENGTH):
        median_h = test_preds["median"][:, h].numpy()
        dir_signal_h = test_preds["direction_signal"][:, h].numpy()
        actuals_h = actual_all[:len(median_h), h].numpy()

        mae = float(np.abs(median_h - actuals_h).mean())
        rmse = float(np.sqrt(((median_h - actuals_h) ** 2).mean()))
        threshold = thresholds[f"horizon_{h+1}"]
        preds_thresholded = (dir_signal_h > threshold).astype(float)
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

    # --- 特徴量重要度 & Attention 重み保存 ---
    print("=== 特徴量重要度・Attention 抽出 ===")
    best_model = models[0]
    raw_out = best_model.predict(test_loader, mode="raw", return_x=True)
    interpretation = best_model.interpret_output(raw_out.output, reduction="mean")

    # 特徴量重要度: encoder_variables を保存
    figs = best_model.plot_interpretation(interpretation)
    enc_ax = figs["encoder_variables"].get_axes()[0]
    enc_names = [t.get_text() for t in enc_ax.get_yticklabels()]
    enc_values = [bar.get_width() for bar in enc_ax.patches]
    feature_importance = dict(zip(enc_names, enc_values))
    plt.close("all")

    importance_path = ARTIFACT_DIR / "feature_importance.json"
    with open(importance_path, "w") as f:
        json.dump(feature_importance, f, indent=2)
    print(f"  Feature importance saved to {importance_path}")

    # Attention 重み: 各ホライゾンごとの attention を 2D 行列で保存
    # 各ホライゾンは encoder + 前の decoder step に attend するため長さが異なる
    # 最大長に合わせてゼロパディングし、ステップラベルも対応させる
    decoder_steps = [f"t+{h+1}" for h in range(PREDICTION_LENGTH)]
    max_len = ENCODER_LENGTH + PREDICTION_LENGTH - 1  # 最後のホライゾンの長さ
    step_labels = [f"t-{ENCODER_LENGTH - i}" for i in range(ENCODER_LENGTH)]
    step_labels += [f"t+{h+1}" for h in range(PREDICTION_LENGTH - 1)]

    attn_rows = []
    for h in range(PREDICTION_LENGTH):
        interp_h = best_model.interpret_output(raw_out.output, reduction="mean", attention_prediction_horizon=h)
        row = interp_h["attention"].tolist()
        # 最大長に合わせてゼロパディング
        row += [0.0] * (max_len - len(row))
        attn_rows.append(row)

    attn_data = {
        "weights": attn_rows,
        "encoder_steps": step_labels,
        "decoder_steps": decoder_steps,
    }

    attn_path = ARTIFACT_DIR / "attention_weights.json"
    with open(attn_path, "w") as f:
        json.dump(attn_data, f, indent=2)
    print(f"  Attention weights saved to {attn_path}")

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
