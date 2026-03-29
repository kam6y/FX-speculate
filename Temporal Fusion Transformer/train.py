"""TFT USD/JPY 学習パイプライン

実行:
    uv run python train.py                # 通常学習
    uv run python train.py --optuna       # Optunaハイパラチューニング付き
    uv run python train.py --walkforward  # Walk-Forwardバックテスト
    uv run python train.py --deploy       # 学習後にダッシュボードへデプロイ
"""

import argparse
import json
import shutil
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from config import CONFIG, ARTIFACT_DIR, BASE_DIR, DEVICE, PIN_MEMORY
from model import DirectionAwareQuantileLoss
from data import (
    prepare_data,
    create_datasets,
    build_unknown_reals,
    tsds_kwargs,
    UNKNOWN_CATEGORICALS,
)


# ===================================================================
# 学習
# ===================================================================
def train_tft(
    training: TimeSeriesDataSet,
    validation: TimeSeriesDataSet,
    config: dict,
    max_epochs: int | None = None,
    fold_id: str | None = None,
) -> tuple[TemporalFusionTransformer, pl.Trainer]:
    """TFT モデル学習"""
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
    old_ckpts = list(ckpt_dir_path.glob("*.ckpt"))

    early_stop = EarlyStopping(
        monitor="val_loss", patience=config["PATIENCE"], mode="min", verbose=True
    )
    ckpt = ModelCheckpoint(
        dirpath=str(ckpt_dir_path),
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
        for old in old_ckpts:
            if old.exists() and str(old) != best_path:
                old.unlink()
    else:
        best_model = tft

    return best_model, trainer


def _direction_accuracy(q50: torch.Tensor, actuals: torch.Tensor, horizon: int) -> float:
    """累積リターンベース方向精度"""
    actual_cum = actuals[:, :horizon].sum(dim=1)
    pred_cum = q50[:, :horizon].sum(dim=1)
    return ((actual_cum > 0) == (pred_cum > 0)).float().mean().item()


# ===================================================================
# 評価
# ===================================================================
def evaluate(
    model: TemporalFusionTransformer,
    dataset: TimeSeriesDataSet,
    config: dict,
) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """モデル評価: MAE / RMSE / 方向精度 / 分位点カバレッジ"""
    dl = dataset.to_dataloader(
        train=False, batch_size=config["BATCH_SIZE"], num_workers=0,
        pin_memory=PIN_MEMORY,
    )

    preds = model.predict(dl, mode="quantiles")
    actuals_list = []
    enc_last_list = []
    for x, y in dl:
        actuals_list.append(y[0].to(DEVICE, non_blocking=True))
        if "encoder_target" in x:
            enc_last_list.append(x["encoder_target"][:, -1].to(DEVICE, non_blocking=True))

    actuals = torch.cat(actuals_list)
    encoder_last = torch.cat(enc_last_list) if enc_last_list else None

    preds = preds.to(DEVICE, non_blocking=True)
    n_q = preds.size(2)
    q_mid = n_q // 2
    q50 = preds[:, :, q_mid]
    q_lo = preds[:, :, 0]
    q_hi = preds[:, :, -1]

    diff = q50 - actuals
    mae = diff.abs().mean().item()
    rmse = (diff ** 2).mean().sqrt().item()

    direction_acc = _direction_accuracy(q50, actuals, 1)
    direction_acc_5d = _direction_accuracy(q50, actuals, 5) if actuals.size(1) >= 5 else direction_acc

    coverage = ((actuals >= q_lo) & (actuals <= q_hi)).float().mean().item()

    metrics: dict = {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "direction_accuracy": round(direction_acc, 4),
        "direction_accuracy_5d": round(direction_acc_5d, 4),
        "quantile_coverage_80": round(coverage, 4),
    }

    for h_idx, h_name in [(0, "1d"), (4, "5d"), (9, "10d"), (19, "20d")]:
        if h_idx < preds.size(1):
            h_mae = (q50[:, h_idx] - actuals[:, h_idx]).abs().mean().item()
            metrics[f"mae_{h_name}"] = round(h_mae, 4)

    return metrics, preds, actuals, encoder_last


# ===================================================================
# トレーディングバックテスト
# ===================================================================
def backtest_trading(
    preds: torch.Tensor,
    actuals: torch.Tensor,
    encoder_last: torch.Tensor | None,
    config: dict,
) -> dict:
    """簡易トレーディングバックテスト"""
    if preds.size(0) < 2:
        return {"error": "insufficient_samples"}

    q_mid = preds.size(2) // 2
    pred_1d = preds[:, 0, q_mid]
    actual_1d = actuals[:, 0]

    cost = (config["SPREAD_PIPS"] + config["SLIPPAGE_PIPS"]) * config["PIP_SIZE"]
    cost_lr = cost / config["PRICE_REFERENCE"]

    signals = torch.zeros_like(pred_1d)
    signals[pred_1d > 0] = 1.0
    signals[pred_1d < 0] = -1.0

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


def compute_backtest_summary(
    preds: torch.Tensor,
    actuals: torch.Tensor,
    pred_time_idx: torch.Tensor,
    df: pd.DataFrame,
    config: dict,
) -> dict:
    """ダッシュボード用のバックテストサマリーを計算 (JSON シリアライズ可能な dict)"""
    n_q = preds.size(2)
    q_mid = n_q // 2
    q50_t = preds[:, :, q_mid]
    q_lo_t = preds[:, :, 0]
    q_hi_t = preds[:, :, -1]

    # Quantile calibration
    quantile_cal = {
        "q10": round(float((actuals <= q_lo_t).float().mean().item()), 4),
        "q50": round(float((actuals <= q50_t).float().mean().item()), 4),
        "q90": round(float((actuals <= q_hi_t).float().mean().item()), 4),
        "coverage_80": round(float(((actuals >= q_lo_t) & (actuals <= q_hi_t)).float().mean().item()), 4),
    }

    # Multi-horizon direction accuracy
    horizon_acc = []
    for h_end, h_name in [(1, "1D"), (2, "2D"), (3, "3D"), (5, "5D"), (10, "10D")]:
        if h_end <= actuals.size(1):
            acc = _direction_accuracy(q50_t, actuals, h_end)
            horizon_acc.append({"horizon": h_name, "accuracy": round(acc, 4)})

    # time_idx -> 日付マッピング
    idx_to_date = dict(zip(df["time_idx"].astype(int), df.index.strftime("%Y-%m-%d")))

    # 重複 time_idx を集約
    ti_np = pred_time_idx.cpu().numpy()
    pred_1d_raw = preds[:, 0, q_mid].cpu().numpy()
    actual_1d_raw = actuals[:, 0].cpu().numpy()
    q10_1d_raw = preds[:, 0, 0].cpu().numpy()
    q90_1d_raw = preds[:, 0, -1].cpu().numpy()

    # 重複 time_idx を集約 (pandas groupby で O(N))
    _agg = pd.DataFrame({
        "ti": ti_np, "pred": pred_1d_raw, "actual": actual_1d_raw,
        "q10": q10_1d_raw, "q90": q90_1d_raw,
    }).groupby("ti").mean()
    unique_ti = _agg.index.values
    pred_1d = _agg["pred"].values
    actual_1d = _agg["actual"].values
    dates = [idx_to_date.get(int(ti), "") for ti in unique_ti]

    # Confidence calibration
    q10_1d = _agg["q10"].values
    q90_1d = _agg["q90"].values
    spread = np.maximum(np.abs(q90_1d - q10_1d), 1e-8)
    conf_scores = np.minimum(np.abs(pred_1d) / spread, 1.0)
    same_sign = ((q10_1d > 0) & (q90_1d > 0)) | ((q10_1d < 0) & (q90_1d < 0))
    conf_scores[same_sign] = np.minimum(conf_scores[same_sign] + 0.3, 1.0)

    correct_dir = (pred_1d > 0) == (actual_1d > 0)
    n_bins = 5
    bin_edges = np.unique(np.percentile(conf_scores, np.linspace(0, 100, n_bins + 1)))
    if len(bin_edges) < 3:
        bin_edges = np.linspace(conf_scores.min(), conf_scores.max() + 1e-9, n_bins + 1)
    n_actual_bins = len(bin_edges) - 1
    bin_idx = np.clip(np.digitize(conf_scores, bin_edges) - 1, 0, n_actual_bins - 1)
    conf_calibration = []
    for b in range(n_actual_bins):
        mask = bin_idx == b
        cnt = int(mask.sum())
        if cnt >= 2:
            conf_calibration.append({
                "bin": f"{bin_edges[b]:.3f}-{bin_edges[b+1]:.3f}",
                "mid": round(float((bin_edges[b] + bin_edges[b+1]) / 2), 4),
                "accuracy": round(float(correct_dir[mask].mean()), 4),
                "avg_conf": round(float(conf_scores[mask].mean()), 4),
                "count": cnt,
            })

    # PnL 計算
    cost = (config["SPREAD_PIPS"] + config["SLIPPAGE_PIPS"]) * config["PIP_SIZE"]
    cost_lr = cost / config.get("PRICE_REFERENCE", 150.0)

    signals = np.zeros_like(pred_1d)
    signals[pred_1d > 0] = 1.0
    signals[pred_1d < 0] = -1.0

    pnl = signals * actual_1d - np.abs(signals) * cost_lr
    cum_pnl = np.cumsum(pnl)

    bh_pnl = actual_1d - cost_lr
    bh_cum = np.cumsum(bh_pnl)

    correct = (pred_1d > 0) == (actual_1d > 0)
    rolling_acc = pd.Series(correct.astype(float)).rolling(60, min_periods=20).mean().values

    # 月次PnL
    df_pnl = pd.DataFrame({"date": pd.to_datetime(dates[:len(pnl)]), "pnl": pnl})
    df_pnl["year"] = df_pnl["date"].dt.year
    df_pnl["month"] = df_pnl["date"].dt.month
    monthly = df_pnl.groupby(["year", "month"])["pnl"].sum().reset_index()
    monthly_data = [
        {"year": int(r["year"]), "month": int(r["month"]), "pnl": round(float(r["pnl"]), 4)}
        for _, r in monthly.iterrows()
    ]

    # トレーディングサマリー
    n_trades = int(np.sum(signals != 0))
    traded_pnl = pnl[signals != 0]
    if n_trades > 0:
        win_rate = float(np.sum(traded_pnl > 0) / n_trades)
        gross_profit = float(traded_pnl[traded_pnl > 0].sum())
        gross_loss = float(np.abs(traded_pnl[traded_pnl < 0]).sum())
        profit_factor = gross_profit / max(gross_loss, 1e-10)
    else:
        win_rate, profit_factor = 0.0, 0.0
    total_pnl = float(cum_pnl[-1]) if len(cum_pnl) > 0 else 0.0
    max_dd = float((cum_pnl - np.maximum.accumulate(cum_pnl)).min()) if len(cum_pnl) > 0 else 0.0
    sharpe = float(pnl.mean() / max(pnl.std(), 1e-10) * (252 ** 0.5))
    mae_1d = float(np.abs(pred_1d - actual_1d).mean())

    return {
        "dates": dates,
        "equity_curve": [round(float(v), 6) for v in cum_pnl],
        "buy_hold": [round(float(v), 6) for v in bh_cum],
        "drawdown": [round(float(v), 6) for v in (cum_pnl - np.maximum.accumulate(cum_pnl))],
        "rolling_accuracy": [round(float(v), 4) if not np.isnan(v) else None for v in rolling_acc],
        "monthly_pnl": monthly_data,
        "daily_pnl": [round(float(v), 6) for v in pnl],
        "signals": [int(s) for s in signals],
        "confidence_calibration": conf_calibration,
        "quantile_calibration": quantile_cal,
        "horizon_accuracy": horizon_acc,
        "summary": {
            "sharpe_ratio": round(sharpe, 4),
            "profit_factor": round(profit_factor, 4),
            "total_pnl": round(total_pnl, 4),
            "max_drawdown": round(max_dd, 4),
            "win_rate": round(win_rate, 4),
            "n_trades": n_trades,
            "mae_1d": round(mae_1d, 4),
            "pred_up_ratio_1d": round(float(np.mean(signals == 1.0)), 4),
        },
    }


# ===================================================================
# モデル解釈
# ===================================================================
def interpret(
    model: TemporalFusionTransformer,
    dataset: TimeSeriesDataSet,
    config: dict,
) -> dict | None:
    """TFT の変数重要度を取得"""
    dl = dataset.to_dataloader(
        train=False, batch_size=config["BATCH_SIZE"], num_workers=0
    )
    try:
        raw_preds = model.predict(dl, mode="raw", return_x=True)
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
# Walk-Forward バックテスト
# ===================================================================
def walk_forward_backtest(df: pd.DataFrame, config: dict) -> dict | None:
    """ウォークフォワードバックテスト"""
    initial_train = config["WF_INITIAL_TRAIN_DAYS"]
    test_window = config["WF_TEST_WINDOW"]
    n = len(df)

    if n < initial_train + test_window:
        print("データ不足: ウォークフォワード不可")
        return None

    unknown_reals = build_unknown_reals(df)
    unknown_cats = [c for c in UNKNOWN_CATEGORICALS if c in df.columns]
    all_metrics: list[dict] = []
    cursor = initial_train
    fold = 0

    while cursor + test_window <= n:
        fold += 1
        full_slice = df.iloc[: cursor + test_window].copy()
        full_slice["time_idx"] = np.arange(len(full_slice))

        val_size = max(int(cursor * 0.1), config["MAX_ENCODER_LENGTH"] + config["MAX_PREDICTION_LENGTH"])
        train_end_idx = cursor - val_size
        if train_end_idx <= 0:
            cursor += test_window
            continue
        train_cutoff_wf = int(full_slice.iloc[train_end_idx - 1]["time_idx"])
        val_cutoff_wf = int(full_slice.iloc[cursor - 1]["time_idx"])
        test_start_wf = int(full_slice.iloc[cursor]["time_idx"])

        try:
            pred_len = min(config["MAX_PREDICTION_LENGTH"], test_window)
            training = TimeSeriesDataSet(
                full_slice[full_slice["time_idx"] <= train_cutoff_wf],
                **tsds_kwargs(config, unknown_reals, unknown_cats, max_prediction_length=pred_len),
            )

            val_ds = TimeSeriesDataSet.from_dataset(
                training,
                full_slice[full_slice["time_idx"] <= val_cutoff_wf],
                min_prediction_idx=train_cutoff_wf + 1,
                stop_randomization=True,
            )

            test_ds = TimeSeriesDataSet.from_dataset(
                training,
                full_slice,
                min_prediction_idx=test_start_wf,
                stop_randomization=True,
            )

            if len(test_ds) == 0 or len(val_ds) == 0:
                cursor += test_window
                continue

            model, _ = train_tft(
                training, val_ds, config, max_epochs=config["WF_FINETUNE_EPOCHS"]
            )
            metrics, _, _, _ = evaluate(model, test_ds, config)
            metrics["fold"] = fold
            metrics["train_end"] = str(df.index[cursor - 1].date())
            all_metrics.append(metrics)

            print(f"  Fold {fold}: MAE={metrics['mae']:.4f}, Dir={metrics['direction_accuracy']:.4f}")

        except Exception as e:
            print(f"  Fold {fold} failed: {e}")

        cursor += test_window
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not all_metrics:
        return None

    numeric_keys = [k for k in all_metrics[0] if isinstance(all_metrics[0].get(k), (int, float))]
    avg = {k: float(np.mean([m[k] for m in all_metrics])) for k in numeric_keys}
    return {"folds": all_metrics, "average": avg}


# ===================================================================
# Optuna ハイパラチューニング
# ===================================================================
def optuna_optimize(df: pd.DataFrame, config: dict) -> dict:
    """Optuna TPE Sampler でハイパーパラメータ最適化"""
    import optuna

    print("\n=== Optuna Hyperparameter Search ===")

    def objective(trial: optuna.Trial) -> float:
        trial_config = config.copy()
        trial_config["HIDDEN_SIZE"] = trial.suggest_categorical("hidden_size", [64, 128, 160, 256])
        trial_config["ATTENTION_HEAD_SIZE"] = trial.suggest_categorical("attention_head_size", [1, 2, 4])
        trial_config["DROPOUT"] = trial.suggest_float("dropout", 0.05, 0.3)
        trial_config["HIDDEN_CONTINUOUS_SIZE"] = trial.suggest_categorical("hidden_continuous_size", [4, 8, 16])
        trial_config["LEARNING_RATE"] = trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True)

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
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config["RANDOM_SEED"]),
    )
    study.optimize(objective, n_trials=config["N_TRIALS"], timeout=config["OPTUNA_TIMEOUT"], show_progress_bar=True)

    print(f"\nBest direction_5d: {study.best_value:.4f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    return study.best_params


# ===================================================================
# メイン
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description="TFT USD/JPY Prediction System")
    parser.add_argument("--optuna", action="store_true", help="Optuna チューニング")
    parser.add_argument("--walkforward", action="store_true", help="Walk-Forward バックテスト")
    parser.add_argument("--deploy", action="store_true", help="ダッシュボードへモデルデプロイ")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    pl.seed_everything(CONFIG["RANDOM_SEED"])

    print(f"Device: {DEVICE}")
    print(f"Artifacts: {ARTIFACT_DIR}\n")

    # 1. データ準備
    df = prepare_data(CONFIG)
    if "close" in df.columns:
        CONFIG["PRICE_REFERENCE"] = float(df["close"].mean())
        print(f"  PRICE_REFERENCE: {CONFIG['PRICE_REFERENCE']:.1f}")

    # 2. Optuna (optional)
    if args.optuna:
        best_params = optuna_optimize(df, CONFIG)
        for key_map in [("hidden_size", "HIDDEN_SIZE"), ("attention_head_size", "ATTENTION_HEAD_SIZE"),
                        ("dropout", "DROPOUT"), ("hidden_continuous_size", "HIDDEN_CONTINUOUS_SIZE"),
                        ("learning_rate", "LEARNING_RATE")]:
            if key_map[0] in best_params:
                CONFIG[key_map[1]] = best_params[key_map[0]]

    # 3. データセット作成
    print("\n=== データセット作成 ===")
    training, validation, test = create_datasets(df, CONFIG)

    # 4. マルチseedアンサンブル学習
    n_seeds = CONFIG["ENSEMBLE_SEEDS"]
    base_seed = CONFIG["RANDOM_SEED"]
    seeds = [base_seed + i * 137 for i in range(n_seeds)]

    print(f"\n=== TFT アンサンブル学習 ({n_seeds} seeds) ===")
    sum_preds = None
    best_model = None
    best_val_dir = -1.0
    all_ckpt_paths: list[str] = []

    for i, seed in enumerate(seeds):
        pl.seed_everything(seed)
        print(f"\n--- Seed {seed} ({i+1}/{n_seeds}) ---")
        model, trainer = train_tft(training, validation, CONFIG, fold_id=f"seed_{i}")
        seed_ckpt = trainer.checkpoint_callback.best_model_path
        if seed_ckpt:
            all_ckpt_paths.append(seed_ckpt)
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

    ens_dir_1d = _direction_accuracy(q50_ens, actuals_dev, 1)
    ens_dir_5d = _direction_accuracy(q50_ens, actuals_dev, 5) if actuals_dev.size(1) >= 5 else ens_dir_1d

    # 5. テスト評価
    print(f"\n=== アンサンブル評価 ({n_seeds} seeds) ===")
    metrics, preds, actuals, encoder_last = evaluate(best_model, test, CONFIG)
    metrics["ensemble_direction_1d"] = round(ens_dir_1d, 4)
    metrics["ensemble_direction_5d"] = round(ens_dir_5d, 4)
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # 6. トレーディングバックテスト
    print("\n=== トレーディングバックテスト ===")
    trade_metrics = backtest_trading(ens_preds.to(DEVICE), actuals_dev, encoder_last, CONFIG)
    for k, v in trade_metrics.items():
        print(f"  {k}: {v}")

    # 7. モデル解釈
    print("\n=== モデル解釈 ===")
    interpret(best_model, test, CONFIG)

    # 8. Walk-Forward (optional)
    wf_results = None
    if args.walkforward:
        print("\n=== Walk-Forward バックテスト ===")
        wf_results = walk_forward_backtest(df, CONFIG)
        if wf_results:
            print("\nWalk-Forward 平均:")
            for k, v in wf_results["average"].items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")

    # 9. アーティファクト保存
    print("\n=== アーティファクト保存 ===")
    all_metrics = {**metrics, **{f"trade_{k}": v for k, v in trade_metrics.items()}}

    with open(ARTIFACT_DIR / "tft_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    dashboard_cache = BASE_DIR / "dashboard" / "backtest_cache.json"
    if dashboard_cache.exists():
        dashboard_cache.unlink()
        print(f"  Cleared: {dashboard_cache}")

    def _serialize_config(cfg: dict) -> dict:
        _secret = {"FRED_API_KEY"}
        return {
            k: v for k, v in cfg.items()
            if k not in _secret and (isinstance(v, (str, int, float, bool, list)) or v is None)
        }

    with open(ARTIFACT_DIR / "tft_config.json", "w") as f:
        json.dump(_serialize_config(CONFIG), f, indent=2, default=str)

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

    # 最適化ログ追記
    opt_log_path = ARTIFACT_DIR / "optimization_log.json"
    opt_log: list = []
    if opt_log_path.exists():
        try:
            with open(opt_log_path) as f:
                opt_log = json.load(f)
        except (json.JSONDecodeError, ValueError):
            opt_log = []

    opt_entry = {
        "iteration": len(opt_log) + 1,
        "timestamp": datetime.now().isoformat(),
        "metrics": all_metrics,
        "config_snapshot": _serialize_config(CONFIG),
    }
    opt_log.append(opt_entry)
    with open(opt_log_path, "w") as f:
        json.dump(opt_log, f, indent=2, ensure_ascii=False, default=str)
    print(f"Optimization log: iteration {opt_entry['iteration']} saved")

    # ダッシュボードへのモデルデプロイ
    if args.deploy:
        deploy_dir = BASE_DIR / "dashboard" / "model"
        deploy_dir.mkdir(parents=True, exist_ok=True)
        for old in deploy_dir.glob("*.ckpt"):
            old.unlink()
        deployed = 0
        for ckpt_p in all_ckpt_paths:
            p = Path(ckpt_p)
            if p.exists():
                dest = deploy_dir / f"seed_{deployed}_{p.name}"
                shutil.copy2(p, dest)
                print(f"  Deployed: {dest.name}")
                deployed += 1
        if deployed > 0:
            for fname in ["tft_metrics.json", "tft_config.json", "feature_schema.json"]:
                src = ARTIFACT_DIR / fname
                if src.exists():
                    shutil.copy2(src, deploy_dir / fname)
            print(f"Deployed {deployed} seed checkpoints to dashboard")
        else:
            print("Warning: No checkpoint found for deploy")

    # lightning_logs クリーンアップ
    for log_dir in [BASE_DIR / "lightning_logs", BASE_DIR / "dashboard" / "lightning_logs"]:
        if log_dir.exists():
            shutil.rmtree(log_dir, ignore_errors=True)

    return all_metrics


if __name__ == "__main__":
    main()
