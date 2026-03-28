# TFT 最適化ループ (Round 9) — Horizon精度チューニング

## 役割
`Temporal Fusion Transformer/train_tft.py` の全Horizon方向精度を改善する。
**train_tft.py のみ編集。1イテレーション = 1変更。**

## 累計 (R1-R8)
R8 で累積方向ペナルティLoss + マルチスケール特徴量を導入。
ベースライン: ens_1d=0.7225, ens_5d=0.3844

### 現在の CONFIG
```
target="log_return", EncoderNormalizer, min_encoder_length=3
HIDDEN_SIZE=192, ATTENTION_HEAD_SIZE=2, DROPOUT=0.15
HIDDEN_CONTINUOUS_SIZE=8, MAX_ENCODER_LENGTH=120, MAX_PREDICTION_LENGTH=10
LR=1e-3, BATCH=64, MAX_EPOCHS=50, PATIENCE=8, GRADIENT_CLIP=1.0
DIRECTION_LOSS_WEIGHT=0.3, CUMULATIVE_DIRECTION_WEIGHT=0.3
weight_decay=5e-3, accumulate_grad_batches=2, reduce_on_plateau_patience=4
ENSEMBLE_SEEDS=3
UNKNOWN_REALS: rsi_14, macd_diff, bb_width, atr_14, ma_dist_5/20/50/200,
               return_5d, return_10d, rsi_5, volatility_10d
OPTIONAL: vix, gold, us10y, interest_rate_diff, sentiment_proxy
```

### R1-R8 全「やるな」リスト（130+ iter で悪化確認）
- CONFIG微調整は全て試行済み（R1-R7, 130 iter）
- 個別ステップ方向ペナルティのみでは Horizon精度は改善しない
- macd, macd_signal, bb_pctb, ema_dist_12, ema_dist_26 は冗長（R2-10で削除済み）
- close を unknown_reals に含めない（target=log_return との冗長性）

## 目標
- ens_1d >= 0.70 を維持
- 全Horizon (direction_1d 〜 direction_10d) で改善
- direction_5d >= 0.45

## 変更候補（優先順位順）

### Phase A: α, β のバランスチューニング
1. α=0.2, β=0.4 (累積ペナルティ重視)
2. α=0.4, β=0.2 (個別ステップ重視)
3. α=0.1, β=0.5 (累積ペナルティ最大化)
4. α=0.3, β=0.5 (両方やや強め)
5. α=0.5, β=0.3 (1d精度重視)

### Phase B: 累積ペナルティの改良
6. 累積ペナルティにHorizon重み付け: 遠いHorizonほど重くする (linear weight)
7. 累積ペナルティのスケーリング: Horizon数で割って正規化
8. Huber-style 累積ペナルティ: 外れ値にロバスト

### Phase C: 訓練パイプライン改修
9. Validation を方向精度で監視 (EarlyStopping の monitor 変更)
10. 訓練時ノイズ注入 (target に微小ガウシアンノイズ)
11. Test-Time Augmentation (エンコーダ入力にノイズ → 複数予測の平均)

## 手順
1. `artifacts/optimization_log.json` の最新エントリを確認し、前回のメトリクスを把握
2. ��の変更候補から1つだけ選んで実装（`# R9-XX:` コメント付与）
3. 実行:
   ```bash
   rm -f "Temporal Fusion Transformer/artifacts/feature_schema.json" && cd "Temporal Fusion Transformer" && uv run python train_tft.py 2>&1 | tail -30
   ```
4. メトリクス確認:
   ```bash
   uv run python -c "import json; m=json.load(open('Temporal Fusion Transformer/artifacts/optimization_log.json'))[-1]['metrics']; print(f'ens_1d={m[\"ensemble_direction_1d\"]}, ens_5d={m[\"ensemble_direction_5d\"]}')"
   ```
5. 詳細Horizon確認:
   ```bash
   uv run python -c "
   import json
   m = json.load(open('Temporal Fusion Transformer/artifacts/optimization_log.json'))[-1]['metrics']
   for k, v in sorted(m.items()):
       if 'direction' in k:
           print(f'  {k}: {v}')
   "
   ```
6. 判断:
   - ens_1d >= 0.70 かつ direction_5d が前回比で改善 → **採用 → コミット**
   - ens_1d < 0.68 → **即リバート** (`git checkout -- "Temporal Fusion Transformer/train_tft.py"`)
   - 横ばい (±0.005) → **リバート** (ノイズの可能性)

## 制約
- train_tft.py のみ、新規パッケージ禁止、GPU/CPU両対応、ログ維持
- エラー時は即revert
- **1イテレーション = 1パラメータ変更**
- 既に試して悪化した変更は再試行しない

## 完了条件
`<promise>HORIZON OPTIMIZED</promise>`:
1. ens_1d >= 0.70 かつ direction_5d >= 0.45
2. 8 iter 連続改善なし
3. 30 iter 到達
