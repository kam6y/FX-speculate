# TFT + XGBoost Stacking Ensemble Design

| Item | Value |
|------|-------|
| Date | 2026-03-22 |
| Status | Approved |
| Parent | TFT_USDJPY予測AI_設計書.md Section 4.1 Phase 4 |

## Problem

TFT with QuantileLoss optimizes price-level prediction (MAE ~3.5 JPY) but achieves only ~49% 1-day direction accuracy across seeds. Direction accuracy is the primary metric for trading profitability.

## Solution

A two-level stacking ensemble. Level 0 (TFT) produces quantile forecasts. Level 1 (XGBClassifier) consumes TFT outputs as meta-features and classifies direction (Up/Down). Walk-Forward OOF generation prevents data leakage.

## Architecture

```
Walk-Forward OOF Generation:
  fold 1: Train[0:750]   -> TFT predict [750:770]   -> meta features
  fold 2: Train[0:770]   -> TFT predict [770:790]   -> meta features
  fold 3: Train[0:790]   -> TFT predict [790:810]   -> meta features
  ...
  concat all fold meta features -> XGB training data

Test Inference:
  TFT (full train) -> test predictions -> meta features -> XGB -> direction signal
```

## Level 0: TFT Meta-Feature Extraction

For each prediction window, extract from TFT quantile output `[N, pred_len, 3]`.
All formulas are element-wise over the N-sample axis; use `np.maximum()` for denominator floors.

| Feature | Formula | Rationale |
|---------|---------|-----------|
| pred_vs_current | (q50_1d - encoder_last) / encoder_last | Predicted relative change. Fallback: if `encoder_last` is None, use last close from fold's training DataFrame |
| q_spread_1d | q90_1d - q10_1d | Prediction uncertainty |
| q_skew | (q90_1d - q50_1d) / np.maximum(q50_1d - q10_1d, 1e-8) | Distribution asymmetry |
| pred_slope | (q50_5d - q50_1d) / 4 | Multi-horizon trend. Guard: if pred_len < 5, set to 0 |
| q50_1d | preds[:, 0, q_mid] | Raw 1-day median |
| q50_5d | preds[:, 4, q_mid] if pred_len >= 5 else preds[:, -1, q_mid] | Raw 5-day median |
| q10_1d | preds[:, 0, 0] | Lower bound |
| q90_1d | preds[:, 0, -1] | Upper bound |

Auxiliary features from original DataFrame (aligned by sample index):

| Feature | Source | Notes |
|---------|--------|-------|
| rsi_14 | Technical indicator | |
| macd_diff | MACD histogram | |
| atr_14 | Volatility | |
| fomc_distance | Event distance (known) | |
| boj_distance | Event distance (known) | |
| market_regime | Regime label | Apply `sklearn.preprocessing.LabelEncoder` before XGB |

Total: ~14 meta-features.

## Level 1: XGBClassifier

- **Target**: binary `1` (Up) / `0` (Down)
- **Label definition**: `y_meta = int(actuals[:, 0] > encoder_last)` where both come from `evaluate()` on the fold's test window. This is the per-sample 1-day direction relative to the encoder's last close.
- **Training data**: Walk-Forward OOF meta-features (~400-600 samples)
- **Hyperparameters**: Optuna TPE search (20 trials)
  - `max_depth`: [3, 8]
  - `n_estimators`: [50, 300]
  - `learning_rate`: [0.01, 0.3] log
  - `subsample`: [0.6, 1.0]
  - `colsample_bytree`: [0.6, 1.0]
  - `scale_pos_weight`: `(y_meta == 0).sum() / (y_meta == 1).sum()` computed once on full OOF set
- **Evaluation metric**: `roc_auc` (more stable than accuracy on small folds)
- **CV**: `TimeSeriesSplit(n_splits=3)` on OOF data

## Walk-Forward OOF Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Initial train window | 750 days | ~3 years |
| Slide step | 20 days | 1 month |
| TFT fine-tune epochs | 20 | Per fold |
| TFT patience | 5 | EarlyStopping per fold |
| Seeds per fold | 1 | Single seed for speed |

Each fold:
1. Train TFT on `[0, cursor]` (with last 10% as validation for EarlyStopping)
2. Predict on `[cursor, cursor+20]` via `evaluate()` (single call, no double inference)
3. Extract meta-features from `preds` (2nd return value of `evaluate()`)
4. Compute `y_meta = int(actuals[:, 0] > encoder_last)`
5. Append (meta_X, meta_y) to OOF buffer
6. Advance cursor by 20
7. Skip fold if remaining data < 5 days (pred_slope guard)

## Data Flow

```
train_ensemble.py main():

  0. Delete feature_schema.json, then call _build_unknown_reals(df_full)
     once on the FULL dataset to establish a stable schema for all folds.

  1. Data collection & feature engineering (reuse train_tft.py functions)

  2. Walk-Forward TFT loop:
     Pre-compute: unknown_reals, unknown_cats from full df
     for each fold:
       - Construct TimeSeriesDataSet directly (using KNOWN_CATEGORICALS,
         KNOWN_REALS, unknown_reals, UNKNOWN_CATEGORICALS from train_tft.py)
       - train_tft() with fine-tune epochs=20, patience=5
       - evaluate() on fold's test window -> (metrics, preds, actuals, encoder_last)
       - Extract meta-features from preds + auxiliary from df
       - Compute y_meta from actuals and encoder_last
       - Append to OOF buffer
       - Clean up: del model, torch.cuda.empty_cache()

  3. Concatenate all OOF -> X_meta [N_oof, 14], y_meta [N_oof]

  4. Optuna XGB search on X_meta, y_meta (TimeSeriesSplit CV, metric=roc_auc)

  5. Train final XGB on full X_meta with best params

  6. Test evaluation:
     - TFT trained on all train data -> predict test via evaluate()
     - Extract same meta-features
     - XGB predict_proba on test meta-features
     - Direction = prob > 0.5

  7. Compute metrics: direction_accuracy, Sharpe, etc.

  8. Save artifacts: ensemble_metrics.json, xgb_model.pkl, meta_features.parquet
```

## File Structure

```
Temporal Fusion Transformer/
  train_tft.py           # Existing - imported as module (NOT modified)
  train_ensemble.py      # New - stacking pipeline
  artifacts/
    ensemble_metrics.json
    xgb_model.pkl
    meta_features.parquet
    ensemble_config.json
    wf_checkpoints/      # Per-fold TFT checkpoints (isolated from main)
```

`train_ensemble.py` imports from `train_tft.py`:
- Functions: `fetch_market_data`, `add_technical_indicators`, `add_calendar_features`, `add_event_distance_features`, `add_holiday_flags`, `add_news_sentiment_proxy`, `add_market_regime`, `preprocess`, `train_tft`, `evaluate`, `_build_unknown_reals`
- Constants: `CONFIG`, `ARTIFACT_DIR`, `DEVICE`, `PIN_MEMORY`, `KNOWN_CATEGORICALS`, `KNOWN_REALS`, `UNKNOWN_CATEGORICALS`

The WF loop constructs `TimeSeriesDataSet` directly (not via `create_datasets()`) because `create_datasets()` uses fixed global train/val/test ratios that don't apply to per-fold splitting.

No modifications to `train_tft.py`.

## Evaluation Metrics

| Metric | Target |
|--------|--------|
| Direction accuracy (1d) | > 0.58 |
| Direction accuracy (5d) | > 0.60 |
| Sharpe Ratio | > 1.0 |
| Max Drawdown | < 15% |

## Risks

| Risk | Mitigation |
|------|------------|
| Walk-Forward TFT is slow (~20 folds x training) | Fine-tune only (20 epochs), not full retrain |
| OOF samples too few for XGB | Use all available folds; XGB is sample-efficient |
| TFT prediction quality varies by seed | Accept variance; XGB learns to handle noisy inputs |
| Overfitting XGB to OOF | Optuna with TimeSeriesSplit CV; roc_auc metric; keep model small |
| feature_schema.json cache corrupts per-fold schema | Delete cache at startup; build once on full df |
| encoder_last is None | Fallback to last close from fold's training DataFrame |
| Checkpoint collision across folds | Use per-fold subdirectory: wf_checkpoints/fold_N/ |

## Execution

```bash
uv run python "Temporal Fusion Transformer/train_ensemble.py"
```

Expected runtime: ~30-60 min (20 WF folds x ~1.5 min TFT fine-tuning + Optuna XGB).
