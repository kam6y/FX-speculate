# TFT + XGBoost Stacking Ensemble - Ralph Loop Agent

USD/JPY 方向予測の TFT + XGBoost スタッキングアンサンブルを**自律的に改善**するエージェント。

## 環境

- 作業ディレクトリ: `Temporal Fusion Transformer/`
- GPU (CUDA + bf16 AMP) を最大限活用
- `train_tft.py`: TFT パイプライン (Level 0)。**CONFIG セクションのみ変更可能**
- `train_ensemble.py`: スタッキングパイプライン (Level 0 + Level 1)。改善の主対象
- `scripts/` 配下の共通モジュールは**変更禁止**

## アーキテクチャ

```
Level 0 (TFT):
  Walk-Forward OOF (750日初期, 20日スライド, pred_len=1)
  → 各fold: TFT fine-tune → 1-step-ahead quantile予測 → メタ特徴量抽出

Level 1 (XGBClassifier):
  メタ特徴量 (TFT quantile出力 + auxiliary) で Up/Down 分類
  → Optuna ハイパラ探索 (20 trials, roc_auc)
  → テスト期間で評価
```

## メタ特徴量 (現在 ~12個)

TFT出力: pred_vs_current, q_spread_1d, q_skew, q50_1d, q10_1d, q90_1d
Auxiliary: rsi_14, macd_diff, atr_14, fomc_distance, boj_distance, market_regime

## 毎イテレーションの手順

### Step 1: ベスト確認

`artifacts/best_ensemble_metrics.json` を読み、現在のベスト direction_accuracy を確認。
`artifacts/history.log` があれば直近の改善履歴を把握。

### Step 2: 改善

`train_ensemble.py` に **1つだけ** 改善を加える。改善の例:

**メタ特徴量の追加/変更:**
- TFT出力から新特徴量 (quantile間の比率、予測の分散、複数ホライズン情報)
- Auxiliary特徴量の追加 (bb_width, log_return, realized_vol, session features)
- 特徴量の削除 (ノイズ源の除去)

**XGBClassifier のチューニング:**
- Optuna探索範囲の変更 (max_depth, n_estimators, learning_rate)
- TimeSeriesSplit の n_splits 調整
- scale_pos_weight の戦略変更
- 閾値最適化 (predict_proba > 0.5 以外)

**Walk-Forward OOF の改善:**
- WF_TEST_WINDOW のステップ幅
- TFT fine-tune エポック数/patience
- OOF データの augmentation

**train_tft.py CONFIG の調整 (CONFIG セクションのみ):**
- HIDDEN_SIZE, ATTENTION_HEAD_SIZE, DROPOUT
- LEARNING_RATE, BATCH_SIZE, GRADIENT_CLIP_VAL
- MAX_ENCODER_LENGTH, MAX_PREDICTION_LENGTH

### Step 3: 実行

```bash
cd "C:/Users/daiya/OneDrive/ドキュメント/FX-speculate/Temporal Fusion Transformer" && uv run python train_ensemble.py
```

### Step 4: メトリクス確認

`artifacts/ensemble_metrics.json` を読み、以下を確認:
- **direction_accuracy** (1日先方向精度, 主要目標)
- **direction_accuracy_5d** (5日先方向精度)
- **roc_auc** (ROC-AUC)
- **up_ratio_predicted** (予測のUp比率, 0.50 に近いほど良い)

### Step 5: ベスト比較・更新

- `best_ensemble_metrics.json` が存在しない(初回) → 現在のメトリクスを保存
- direction_accuracy が改善 → `best_ensemble_metrics.json` を更新、コードをバックアップ
- direction_accuracy が悪化 → 変更をリバート

### Step 6: 履歴記録

`artifacts/history.log` に追記:
```
--- Iteration N ---
Change: [何を変更したか]
Dir_1d: [今回] (Best: [ベスト])
Dir_5d: [今回]
AUC: [今回]
Result: [IMPROVED / REVERTED]
```

## 重要な制約

- `scripts/` 配下は変更禁止
- `data/` 配下は変更禁止
- **テストデータでの学習禁止**: OOF ループは `oof_end = n * (TRAIN_RATIO + VAL_RATIO)` まで
- auxiliary 特徴量は **encoder末端日** (予測時点で既知) の値を使用
- GPU を使うこと
- 各イテレーションで1つの変更のみ

## 過去の実験から学んだこと

- FX 日次データの1日先方向精度は ~0.50 (ランダム) が baseline
- TFT 単体では方向精度 ~0.49-0.52 が限界
- TFT + XGBoost スタッキングで ~0.52 (リーク修正後)
- ハイパラ変更の効果は seed 依存性が高く再現性が低い
- up_ratio_predicted が 0.50 から大きく外れる場合、XGB が偏っている
- 0.7339 はデータリークの産物。フェアな評価では 0.55 が実質天井
