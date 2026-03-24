"""TFT + XGBoost Stacking Ensemble - USD/JPY Direction Prediction

Spec: docs/superpowers/specs/2026-03-22-tft-xgboost-stacking-design.md
Phase 4 of TFT_USDJPY予測AI_設計書.md

Level 0: TFT (quantile forecasts via Walk-Forward OOF)
Level 1: XGBClassifier (direction classification on TFT meta-features)

Usage:
    uv run python "Temporal Fusion Transformer/train_ensemble.py"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import json
import pickle
import warnings

import numpy as np
import pandas as pd
import torch
import optuna
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, accuracy_score
from xgboost import XGBClassifier

from train_tft import (
    fetch_market_data,
    add_technical_indicators,
    add_calendar_features,
    add_event_distance_features,
    add_holiday_flags,
    add_news_sentiment_proxy,
    add_market_regime,
    preprocess,
    train_tft,
    evaluate,
    _build_unknown_reals,
    CONFIG,
    ARTIFACT_DIR,
    DEVICE,
    PIN_MEMORY,
    KNOWN_CATEGORICALS,
    KNOWN_REALS,
    UNKNOWN_CATEGORICALS,
)
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
import lightning.pytorch as pl


# ===================================================================
# 1. Meta-Feature Extraction
# ===================================================================
def extract_meta_features(
    preds: torch.Tensor,
    actuals: torch.Tensor,
    encoder_last: torch.Tensor | None,
    df: pd.DataFrame,
    sample_indices: pd.Index,
) -> tuple[pd.DataFrame, np.ndarray]:
    """TFT evaluate() 出力から XGB 用メタ特徴量を抽出。

    Returns:
        (X_meta DataFrame [N, ~14], y_meta ndarray [N])
    """
    p = preds.detach().cpu().numpy()
    a = actuals.detach().cpu().numpy()
    n = p.shape[0]
    pred_len = p.shape[1]
    q_mid = p.shape[2] // 2

    # --- TFT quantile features ---
    q50_1d = p[:, 0, q_mid]
    q10_1d = p[:, 0, 0]
    q90_1d = p[:, 0, -1]
    q_spread = q90_1d - q10_1d
    lower_spread = q50_1d - q10_1d
    upper_spread = q90_1d - q50_1d
    # quantile crossing (q10 > q50) 時は q_skew=0 として扱い外れ値を防止
    q_skew = np.where(
        lower_spread > 1e-4,
        upper_spread / lower_spread,
        0.0,
    )

    # encoder_last fallback (pytorch-forecasting では通常 None にならない)
    if encoder_last is not None:
        enc = encoder_last.detach().cpu().numpy()
    else:
        warnings.warn("encoder_last is None: pred_vs_current/y_meta may be unreliable")
        enc = a[:, 0]

    pred_vs_current = (q50_1d - enc) / np.maximum(np.abs(enc), 1e-8)

    meta = pd.DataFrame(
        {
            "pred_vs_current": pred_vs_current,
            "q_spread_1d": q_spread,
            "q_skew": q_skew,
            "q50_1d": q50_1d,
            "q10_1d": q10_1d,
            "q90_1d": q90_1d,
        }
    )

    # --- Target: 1-day direction (inverted: 1=Down, 0=Up) ---
    # XGB learns inverse correlation better; invert labels at training
    y_meta = (a[:, 0] <= enc).astype(np.int32)

    return meta, y_meta


# ===================================================================
# 2. Walk-Forward OOF Generation
# ===================================================================
def walk_forward_oof(
    df: pd.DataFrame,
    config: dict,
    unknown_reals: list[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    """Walk-Forward で TFT OOF 予測を蓄積しメタ特徴量を生成。

    OOF は train+val 期間のみ (テスト期間はリーク防止のため除外)。
    """
    initial_train = config.get("WF_INITIAL_TRAIN_DAYS", 750)
    step = config.get("WF_TEST_WINDOW", 20)
    n = len(df)

    # OOF はテスト期間の手前まで (リーク防止)
    oof_end = int(n * (config["TRAIN_RATIO"] + config["VAL_RATIO"]))

    unknown_cats = [c for c in UNKNOWN_CATEGORICALS if c in df.columns]

    oof_X: list[pd.DataFrame] = []
    oof_y: list[np.ndarray] = []
    cursor = initial_train
    fold = 0

    while cursor + step <= oof_end:
        remaining = oof_end - cursor
        if remaining < 5:
            break
        fold += 1
        actual_step = min(step, remaining)

        full_slice = df.iloc[: cursor + actual_step].copy()
        full_slice["time_idx"] = np.arange(len(full_slice))

        # Train/val split: last 10% of train as validation
        val_size = max(
            int(cursor * 0.1),
            config["MAX_ENCODER_LENGTH"] + config["MAX_PREDICTION_LENGTH"],
        )
        train_end_idx = cursor - val_size
        train_cutoff = int(full_slice.iloc[train_end_idx - 1]["time_idx"])
        val_cutoff = int(full_slice.iloc[cursor - 1]["time_idx"])
        test_start = int(full_slice.iloc[cursor]["time_idx"])
        # pred_len=1: 各日が独立サンプルになりOOFデータ量を最大化
        pred_len = 1

        try:
            training = TimeSeriesDataSet(
                full_slice[full_slice["time_idx"] <= train_cutoff],
                time_idx="time_idx",
                target="close",
                group_ids=["group_id"],
                max_encoder_length=config["MAX_ENCODER_LENGTH"],
                max_prediction_length=pred_len,
                static_categoricals=["group_id"],
                time_varying_known_categoricals=KNOWN_CATEGORICALS,
                time_varying_known_reals=KNOWN_REALS,
                time_varying_unknown_categoricals=unknown_cats,
                time_varying_unknown_reals=unknown_reals,
                target_normalizer=GroupNormalizer(groups=["group_id"]),
                add_relative_time_idx=True,
                add_target_scales=True,
                add_encoder_length=True,
            )

            val_ds = TimeSeriesDataSet.from_dataset(
                training,
                full_slice[full_slice["time_idx"] <= val_cutoff],
                min_prediction_idx=train_cutoff + 1,
                stop_randomization=True,
            )

            test_ds = TimeSeriesDataSet.from_dataset(
                training,
                full_slice,
                min_prediction_idx=test_start,
                stop_randomization=True,
            )

            if len(test_ds) == 0 or len(val_ds) == 0:
                cursor += step
                continue

            model, _ = train_tft(
                training,
                val_ds,
                config,
                max_epochs=config.get("WF_FINETUNE_EPOCHS", 20),
            )
            metrics, preds, actuals, encoder_last = evaluate(model, test_ds, config)

            test_indices = df.index[cursor : cursor + actual_step]
            X_fold, y_fold = extract_meta_features(
                preds, actuals, encoder_last, df, test_indices
            )
            oof_X.append(X_fold)
            oof_y.append(y_fold)

            print(
                f"  Fold {fold}: samples={len(X_fold)}, "
                f"dir={metrics.get('direction_accuracy', 0):.3f}"
            )

            del model
        except Exception as e:
            print(f"  Fold {fold} failed: {e}")

        cursor += step
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not oof_X:
        raise RuntimeError("No OOF data generated")

    X_all = pd.concat(oof_X, ignore_index=True)
    y_all = np.concatenate(oof_y)
    print(f"\nOOF total: {len(X_all)} samples, up_ratio={y_all.mean():.3f}")
    return X_all, y_all


# ===================================================================
# 3. XGB Classifier Training (Optuna)
# ===================================================================
def train_xgb_classifier(
    X_oof: pd.DataFrame,
    y_oof: np.ndarray,
    config: dict,
) -> tuple[XGBClassifier, LabelEncoder | None]:
    """OOF メタ特徴量で XGBClassifier を学習 (Optuna 付き)。"""
    le = None
    X_train = X_oof.copy()
    if "market_regime" in X_train.columns:
        le = LabelEncoder()
        X_train["market_regime"] = le.fit_transform(
            X_train["market_regime"].fillna("range")
        )
    X_train = X_train.fillna(0)

    scale_pos = float((y_oof == 0).sum() / max((y_oof == 1).sum(), 1))
    use_gpu = torch.cuda.is_available()

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "scale_pos_weight": scale_pos,
            "eval_metric": "logloss",
            "tree_method": "hist",
            "device": "cuda" if use_gpu else "cpu",
            "random_state": config["RANDOM_SEED"],
        }

        scores = []
        tscv = TimeSeriesSplit(n_splits=5)
        for train_idx, val_idx in tscv.split(X_train):
            clf = XGBClassifier(**params)
            clf.fit(X_train.iloc[train_idx], y_oof[train_idx])
            y_prob = clf.predict_proba(X_train.iloc[val_idx])[:, 1]
            try:
                auc = roc_auc_score(y_oof[val_idx], y_prob)
            except ValueError:
                auc = 0.5
            scores.append(auc)
        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config["RANDOM_SEED"]),
    )
    study.optimize(objective, n_trials=20, timeout=300)

    print(f"  Optuna best AUC: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")

    best_params = {
        **study.best_params,
        "scale_pos_weight": scale_pos,
        "eval_metric": "logloss",
        "tree_method": "hist",
        "device": "cuda" if use_gpu else "cpu",
        "random_state": config["RANDOM_SEED"],
    }
    final_clf = XGBClassifier(**best_params)
    final_clf.fit(X_train, y_oof)

    return final_clf, le


# ===================================================================
# 4. Ensemble Evaluation
# ===================================================================
def evaluate_ensemble(
    xgb_model: XGBClassifier,
    preds: torch.Tensor,
    actuals: torch.Tensor,
    encoder_last: torch.Tensor | None,
    df: pd.DataFrame,
    sample_indices: pd.Index,
    label_encoder: LabelEncoder | None,
) -> dict:
    """アンサンブル評価: XGB の方向予測精度を計算。"""
    X_test, y_test = extract_meta_features(
        preds, actuals, encoder_last, df, sample_indices
    )

    X_eval = X_test.copy()
    if label_encoder is not None and "market_regime" in X_eval.columns:
        known = set(label_encoder.classes_)
        X_eval["market_regime"] = X_eval["market_regime"].map(
            lambda v: v if v in known else label_encoder.classes_[0]
        )
        X_eval["market_regime"] = label_encoder.transform(X_eval["market_regime"])
    X_eval = X_eval.fillna(0)

    y_prob = xgb_model.predict_proba(X_eval)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)

    dir_acc = float(accuracy_score(y_test, y_pred))

    # 5-day direction
    a_np = actuals.detach().cpu().numpy()
    enc_np = (
        encoder_last.detach().cpu().numpy()
        if encoder_last is not None
        else a_np[:, 0]
    )
    if actuals.size(1) >= 5:
        actual_5d = (a_np[:, 4] <= enc_np).astype(int)  # inverted to match training labels
        dir_acc_5d = float(accuracy_score(actual_5d, y_pred))
    else:
        dir_acc_5d = dir_acc

    try:
        auc = float(roc_auc_score(y_test, y_prob))
    except ValueError:
        auc = 0.5

    return {
        "direction_accuracy": round(dir_acc, 4),
        "direction_accuracy_5d": round(dir_acc_5d, 4),
        "roc_auc": round(auc, 4),
        "n_samples": len(y_test),
        "up_ratio_actual": round(float(y_test.mean()), 4),
        "up_ratio_predicted": round(float(y_pred.mean()), 4),
    }


# ===================================================================
# 5. Main Pipeline
# ===================================================================
def main():
    warnings.filterwarnings("ignore")
    pl.seed_everything(CONFIG["RANDOM_SEED"])

    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"Artifacts: {ARTIFACT_DIR}\n")

    # ── 1. Data Collection & Feature Engineering ──
    print("=== 1. Data Collection ===")
    df = fetch_market_data(CONFIG)

    print("\n=== 2. Feature Engineering ===")
    df = add_technical_indicators(df)
    df = add_calendar_features(df)
    df = add_event_distance_features(df)
    df = add_holiday_flags(df)
    df = add_news_sentiment_proxy(df)
    df = add_market_regime(df)

    print("\n=== 3. Preprocessing ===")
    df = preprocess(df)

    # ── 2. Stable feature schema ──
    schema_path = ARTIFACT_DIR / "feature_schema.json"
    if schema_path.exists():
        schema_path.unlink()
    unknown_reals = _build_unknown_reals(df)

    # ── 3. Walk-Forward OOF ──
    print("\n=== 4. Walk-Forward OOF Generation ===")
    wf_config = {
        **CONFIG,
        "MAX_EPOCHS": CONFIG.get("WF_FINETUNE_EPOCHS", 20),
        "PATIENCE": 5,
        "DROPOUT": 0.1,  # lower dropout for better OOF predictions
        "LEARNING_RATE": 5e-4,  # lower LR for finer convergence
    }
    X_oof, y_oof = walk_forward_oof(df, wf_config, unknown_reals)

    X_oof.to_parquet(ARTIFACT_DIR / "meta_features.parquet", index=False)

    # ── 4. XGB Training ──
    print("\n=== 5. XGB Classifier Training ===")
    xgb_model, label_encoder = train_xgb_classifier(X_oof, y_oof, CONFIG)

    with open(ARTIFACT_DIR / "xgb_model.pkl", "wb") as f:
        pickle.dump({"model": xgb_model, "label_encoder": label_encoder}, f)

    # ── 5. Final TFT -> Test ──
    print("\n=== 6. Final TFT Training + Test Prediction ===")
    from train_tft import create_datasets

    training, validation, test = create_datasets(df, CONFIG)
    tft_model, _ = train_tft(training, validation, CONFIG)
    _, preds, actuals, encoder_last = evaluate(tft_model, test, CONFIG)

    n = len(df)
    val_end = int(n * (CONFIG["TRAIN_RATIO"] + CONFIG["VAL_RATIO"]))
    test_indices = df.index[val_end:]

    # ── 6. Ensemble Evaluation ──
    print("\n=== 7. Ensemble Evaluation ===")
    ens_metrics = evaluate_ensemble(
        xgb_model, preds, actuals, encoder_last, df, test_indices, label_encoder
    )
    for k, v in ens_metrics.items():
        print(f"  {k}: {v}")

    # ── 7. Save Artifacts ──
    print("\n=== Artifacts ===")
    with open(ARTIFACT_DIR / "ensemble_metrics.json", "w") as f:
        json.dump(ens_metrics, f, indent=2)

    with open(ARTIFACT_DIR / "ensemble_config.json", "w") as f:
        json.dump(
            {"xgb_params": xgb_model.get_params(), "n_oof_samples": len(X_oof)},
            f,
            indent=2,
            default=str,
        )

    print(f"Saved to {ARTIFACT_DIR}")
    print(f"\n{'='*50}")
    print(f"Direction (1d): {ens_metrics['direction_accuracy']:.4f}")
    print(f"Direction (5d): {ens_metrics['direction_accuracy_5d']:.4f}")
    print(f"ROC-AUC: {ens_metrics['roc_auc']:.4f}")
    print(f"{'='*50}")

    return ens_metrics


if __name__ == "__main__":
    main()
