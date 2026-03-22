"""
Ralph Loop — 自律型USD/JPY予測モデル構築スクリプト

GPU (CUDA + AMP) を最大限活用したTransformerベースの予測パイプライン。
エージェントはこのファイルの CONFIG セクションとモデルアーキテクチャを
自由に変更して改善を図る。

実行: uv run python train.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import math
import pickle
import warnings

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight

from scripts.data_loader import load_data
from scripts.evaluation import (
    DEFAULT_CONFIG,
    build_live_filter,
    compute_metrics,
    predict_with_thresholds,
    run_backtest,
)
from scripts.features import (
    create_target,
    generate_features,
    get_feature_columns,
    prepare_ohlcv,
)

# ──────────────────────────────────────────────
# GPU セットアップ
# ──────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# ===================================================================
# CONFIG — エージェントが変更するセクション
# ===================================================================
CONFIG = {
    **DEFAULT_CONFIG,
    # ターゲット定義
    "PREDICT_HORIZON": 15,
    "THRESHOLD_PIPS_MIN": 2,
    "THRESHOLD_PIPS_MAX": 10,
    "THRESHOLD_PIPS_DEFAULT": 5,
    # 確率閾値
    "PROB_THRESHOLD_MIN": 0.40,
    "PROB_THRESHOLD_MAX": 0.80,
    # データ分割
    "TRAIN_RATIO": 0.6,
    "VAL_RATIO": 0.2,
    # Optuna
    "N_TRIALS": 10,
    "OPTUNA_TIMEOUT": 2400,
    "OPTUNA_SUBSAMPLE": 0.3,
    # 学習
    "MAX_EPOCHS_SEARCH": 20,
    "PATIENCE_SEARCH": 5,
    "MAX_EPOCHS_FINAL": 200,
    "PATIENCE_FINAL": 20,
    # 再現性
    "RANDOM_SEED": 99,
    # Transformer探索範囲
    "WINDOW_SIZE_MIN": 30,
    "WINDOW_SIZE_MAX": 120,
    "WINDOW_SIZE_STEP": 10,
    "D_MODEL_CHOICES": [32, 64],
    "NHEAD_CHOICES": [2, 4],
    "NUM_LAYERS_MIN": 1,
    "NUM_LAYERS_MAX": 2,
    "DIM_FF_CHOICES": [64, 128],
    "DROPOUT_MIN": 0.1,
    "DROPOUT_MAX": 0.4,
    "LR_MIN": 1e-4,
    "LR_MAX": 1e-3,
    "BATCH_SIZE_CHOICES": [256, 512],
}


# ===================================================================
# GPU上のデータセット — DataLoader不要、全データVRAM上に保持
# ===================================================================
class GPUWindowDataset:
    """全データをGPU VRAM上に保持し、ウィンドウ生成もGPU上で実行。
    CPU⇔GPU転送が一切発生しないため、GPU使用率が大幅に向上する。
    メモリ: 880K samples × 99 features × 4 bytes ≈ 350MB (16GB VRAMで余裕)
    """

    def __init__(self, X: pd.DataFrame, y: pd.Series, window_size: int, device: torch.device):
        self.X = torch.tensor(X.values, dtype=torch.float32, device=device)
        self.y = torch.tensor(y.values, dtype=torch.long, device=device)
        self.window_size = window_size
        self.n_samples = len(X) - window_size
        self.offsets = torch.arange(window_size, device=device)

    def __len__(self):
        return self.n_samples

    def get_batch(self, indices: torch.Tensor):
        """GPU上でバッチ取得。indices は GPU tensor。"""
        win_idx = indices.unsqueeze(1) + self.offsets.unsqueeze(0)
        windows = self.X[win_idx]  # [B, W, F] — GPU advanced indexing
        labels = self.y[indices + self.window_size]
        return windows, labels

    def shuffled_batches(self, batch_size: int):
        """シャッフルされたバッチのイテレータ（学習用）"""
        perm = torch.randperm(self.n_samples, device=self.X.device)
        for i in range(0, self.n_samples, batch_size):
            yield self.get_batch(perm[i : i + batch_size])

    def sequential_batches(self, batch_size: int):
        """順次バッチのイテレータ（推論用）"""
        for i in range(0, self.n_samples, batch_size):
            end = min(i + batch_size, self.n_samples)
            indices = torch.arange(i, end, device=self.X.device)
            yield self.get_batch(indices)


# ===================================================================
# モデルアーキテクチャ — エージェントが変更可能
# ===================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class FXTransformer(nn.Module):
    def __init__(
        self, n_features, d_model, nhead, num_layers, dim_feedforward, dropout, num_classes=3
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # ウィンドウ単位の標準化（GPU上、float32強制でAMP安全）
        x_f32 = x.float()
        mean = x_f32.mean(dim=1, keepdim=True)
        std = x_f32.std(dim=1, keepdim=True, unbiased=False).clamp(min=1e-6)
        x = ((x_f32 - mean) / std).to(x.dtype)

        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = x[:, -1, :]
        x = self.dropout(x)
        return self.classifier(x)


# ===================================================================
# 学習・推論
# ===================================================================
def train_model(
    X_train, y_train, X_val, y_val,
    window_size, d_model, nhead, num_layers, dim_feedforward,
    dropout, lr, batch_size, max_epochs, patience,
):
    train_ds = GPUWindowDataset(X_train, y_train, window_size, DEVICE)
    val_ds = GPUWindowDataset(X_val, y_val, window_size, DEVICE)

    n_features = X_train.shape[1]
    model = FXTransformer(
        n_features, d_model, nhead, num_layers, dim_feedforward, dropout
    ).to(DEVICE)

    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        for X_batch, y_batch in train_ds.shuffled_batches(batch_size):
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                output = model(X_batch)
                loss = criterion(output, y_batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()

        model.eval()
        val_loss = 0
        n_batches = 0
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=USE_AMP):
            for X_batch, y_batch in val_ds.sequential_batches(batch_size):
                output = model(X_batch)
                val_loss += criterion(output, y_batch).item()
                n_batches += 1
        val_loss /= max(n_batches, 1)

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


def predict_model(model, X, y_dummy, window_size, batch_size=512):
    ds = GPUWindowDataset(X, y_dummy, window_size, DEVICE)
    model.eval()
    all_probs = []
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=USE_AMP):
        for X_batch, _ in ds.sequential_batches(batch_size):
            output = model(X_batch)
            probs = torch.softmax(output, dim=1).cpu().numpy()
            all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


# ===================================================================
# メインパイプライン
# ===================================================================
def main(seed=None):
    warnings.filterwarnings("ignore")
    s = seed if seed is not None else CONFIG["RANDOM_SEED"]
    np.random.seed(s)
    torch.manual_seed(s)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(s)
    print(f"\n{'='*50}")
    print(f"SEED: {s}")
    print(f"{'='*50}")

    print(f"Device: {DEVICE}, AMP: {USE_AMP}")
    print(f"Artifacts: {ARTIFACT_DIR}")

    # ── 1. データ読み込み ──
    data_dir = str(Path(__file__).parent.parent / "data")
    df = load_data(data_dir)
    df, price_cols = prepare_ohlcv(df)
    df_features = generate_features(df, price_cols)

    feature_cols = get_feature_columns(df_features)
    for col in ["open", "high", "low", "close"]:
        if col not in feature_cols:
            feature_cols.append(col)

    future_returns_pips = (
        df_features["close"].shift(-CONFIG["PREDICT_HORIZON"]) - df_features["close"]
    ) / CONFIG["PIP_SIZE"]

    X_all = df_features[feature_cols].copy()
    valid_indices = X_all.dropna().index
    valid_indices = valid_indices[: -CONFIG["PREDICT_HORIZON"]]
    X_all = X_all.loc[valid_indices]
    fr_valid = future_returns_pips.loc[valid_indices]

    print(f"Features: {len(feature_cols)}, Samples: {len(X_all)}")

    # ── 2. データ分割 ──
    max_window = CONFIG["WINDOW_SIZE_MAX"]
    n = len(X_all)
    train_end = int(n * CONFIG["TRAIN_RATIO"])
    val_end = int(n * (CONFIG["TRAIN_RATIO"] + CONFIG["VAL_RATIO"]))
    gap = CONFIG["PREDICT_HORIZON"] + max_window

    X_train = X_all.iloc[:train_end]
    val_start = min(train_end + gap, n)
    X_val = X_all.iloc[val_start:val_end]
    test_start = min(val_end + gap, n)
    X_test = X_all.iloc[test_start:]

    fr_train = fr_valid.loc[X_train.index]
    fr_val = fr_valid.loc[X_val.index]
    fr_test = fr_valid.loc[X_test.index]

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # ── ATR閾値（Optuna内でも使用） ──
    if "volatility_atr" in df_features.columns:
        atr_series = df_features.loc[X_train.index, "volatility_atr"]
        atr_threshold = float(atr_series.quantile(CONFIG["ATR_PERCENTILE"] / 100))
    else:
        atr_threshold = 0.0

    # ── 3. 固定パラメータ（Optunaバイパス） ──
    bp = {
        "threshold_pips": 3.0,
        "prob_threshold": 0.48,
        "window_size": 50,
        "d_model": 32,
        "nhead": 2,
        "num_layers": 1,
        "dim_feedforward": 128,
        "dropout": 0.25,
        "lr": 3e-4,
        "batch_size": 512,
    }
    print("\n=== Fixed Parameters (no Optuna) ===")
    for k, v in bp.items():
        print(f"  {k}: {v}")

    # ── 4. フルデータで最終学習 ──
    print("\n=== Final Training (full data) ===")
    best_threshold_pips = bp["threshold_pips"]
    best_window_size = bp["window_size"]

    y_train_final = create_target(
        df_features.loc[X_train.index], best_threshold_pips, CONFIG["PREDICT_HORIZON"]
    )
    y_val_final = create_target(
        df_features.loc[X_val.index], best_threshold_pips, CONFIG["PREDICT_HORIZON"]
    )
    y_test_final = create_target(
        df_features.loc[X_test.index], best_threshold_pips, CONFIG["PREDICT_HORIZON"]
    )

    model = train_model(
        X_train, y_train_final.loc[X_train.index],
        X_val, y_val_final.loc[X_val.index],
        best_window_size, bp["d_model"], bp["nhead"], bp["num_layers"],
        bp["dim_feedforward"], bp["dropout"], bp["lr"], bp["batch_size"],
        CONFIG["MAX_EPOCHS_FINAL"], CONFIG["PATIENCE_FINAL"],
    )

    # ── 5. 確率閾値 ──
    threshold_buy = bp["prob_threshold"]
    threshold_sell = bp["prob_threshold"]
    print(f"Thresholds - Buy: {threshold_buy:.4f}, Sell: {threshold_sell:.4f}")

    # ── 7. テストセットでバックテスト ──
    print("\n=== Test Set Backtest ===")
    probs_test = predict_model(
        model, X_test, y_test_final.loc[X_test.index], best_window_size
    )
    test_indices = X_test.index[best_window_size:]
    preds_test = predict_with_thresholds(probs_test, threshold_buy, threshold_sell)

    backtest_config = {**CONFIG, "atr_threshold": atr_threshold}
    result = run_backtest(preds_test, test_indices, df, df_features, backtest_config)
    metrics = compute_metrics(result, CONFIG)

    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # ── 8. アーティファクト保存 ──
    metrics_out = {}
    for k, v in metrics.items():
        metrics_out[k] = float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v

    with open(ARTIFACT_DIR / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    torch.save(model.state_dict(), ARTIFACT_DIR / "model.pt")

    config_save = CONFIG.copy()
    config_save["THRESHOLD_PIPS"] = best_threshold_pips
    config_save["THRESHOLD_BUY"] = threshold_buy
    config_save["THRESHOLD_SELL"] = threshold_sell
    config_save["ATR_THRESHOLD"] = atr_threshold
    config_save["WINDOW_SIZE"] = best_window_size
    config_save["model_params"] = {
        "n_features": len(feature_cols),
        "d_model": bp["d_model"],
        "nhead": bp["nhead"],
        "num_layers": bp["num_layers"],
        "dim_feedforward": bp["dim_feedforward"],
        "dropout": bp["dropout"],
    }
    config_save["feature_cols"] = feature_cols
    config_save["best_params"] = bp

    with open(ARTIFACT_DIR / "config.pkl", "wb") as f:
        pickle.dump(config_save, f)

    print(f"\nArtifacts saved to {ARTIFACT_DIR}")
    print(f"Sharpe Ratio: {metrics_out['sharpe_ratio']:.4f}")

    return metrics_out


if __name__ == "__main__":
    main()
