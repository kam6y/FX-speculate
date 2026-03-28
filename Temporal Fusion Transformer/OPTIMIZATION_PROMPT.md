# TFT 最適化ループ (Round 8) — コード改修による構造的改善

## 役割
`Temporal Fusion Transformer/train_tft.py` の方向予測精度をさらに改善する。
**train_tft.py のみ編集。1イテレーション = 1変更。**

## 累計 (R1-R7, 130 iter)
ベスト: ens_1d=**0.6766**, ens_5d=**0.6826**
R6 の target=log_return 転換が最大の改善。R7 では min_encoder_length=15 が効いた。
CONFIG パラメータは完全に収束済み。

### 現在の CONFIG
```
target="log_return", EncoderNormalizer, min_encoder_length=15
HIDDEN_SIZE=192, ATTENTION_HEAD_SIZE=2, DROPOUT=0.15
HIDDEN_CONTINUOUS_SIZE=8, MAX_ENCODER_LENGTH=120, MAX_PREDICTION_LENGTH=10
LR=1e-3, BATCH=64, MAX_EPOCHS=50, PATIENCE=8, GRADIENT_CLIP=1.0
DIRECTION_LOSS_WEIGHT=0.5, weight_decay=5e-3
accumulate_grad_batches=2, reduce_on_plateau_patience=4
ENSEMBLE_SEEDS=3
UNKNOWN_REALS: rsi_14, macd_diff, bb_width, atr_14, ma_dist_5/20/50/200
OPTIONAL: vix, gold, us10y, interest_rate_diff, sentiment_proxy
```

### R1-R7 全「やるな」リスト（130 iter で悪化確認）
CONFIG微調整は全て試行済み。R8 では**コードの書き換え**のみ。

## 目標
ens_1d >= 0.70 (5d累積方向精度は参考値。デイリーモデルのため1d精度が最重要)

## 変更候補（コード改修のみ、CONFIG変更は全て試行済み）

### Phase A: 訓練パイプライン改修
1. **訓練時データ拡張（ノイズ注入）**: DataLoader からバッチを取得した後、target(log_return) に微小ガウシアンノイズを加える。train_tft() 内でカスタム training loop を実装。
   ```python
   # train_dl のラッパーでノイズ注入
   class NoisyDataLoader:
       def __init__(self, dl, noise_std=0.001):
           self.dl = dl
           self.noise_std = noise_std
       def __iter__(self):
           for x, y in self.dl:
               y = (y[0] + torch.randn_like(y[0]) * self.noise_std, *y[1:])
               yield x, y
       def __len__(self): return len(self.dl)
   ```

2. **Validation を方向精度で監視**: EarlyStopping を `monitor="val_loss"` ではなく、カスタムメトリクスの方向精度で監視。
   - TFT の `training_step` / `validation_step` を override するサブクラスを作成
   - または callback で val_direction_accuracy をログに記録

3. **Mixup データ拡張**: 2つのサンプルの特徴量と target を混合して新たなサンプルを生成。

### Phase B: アンサンブル改修
4. **Top-K seed 選択**: 3 seed のうち val_dir が最低のモデルを除外し、残り2つだけでアンサンブル。
5. **Prediction-time augmentation**: テスト時にエンコーダ入力に微小ノイズを加えて複数回予測し、その平均を取る(Test-Time Augmentation)。

### Phase C: 特徴量エンジニアリング（生成ロジック変更）
6. **RSI を 0-1 正規化**: 現在の rsi_14 は 0-100 スケール。0-1 に正規化して他特徴量とスケールを合わせる。
7. **ATR を close で正規化**: atr_14 を atr_14/close に変更（相対ボラティリティ）。
8. **全特徴量の z-score 正規化**: preprocess() で訓練データの mean/std で全 unknown_reals を z-score 正規化。

### Phase D: 損失関数改修（target=log_return 特化）
9. **方向ペナルティを符号ベースに変更**: 現在は「連続タイムステップの差分」ベース。log_return 空間では `pred > 0` vs `target > 0` の不一致にペナルティを与える方が直接的。
10. **Huber-style quantile loss**: 外れ値に対するロバスト性を高める。QuantileLoss の `loss()` 内で小さな値は L2、大きな値は L1 にする。

### Phase E: min_encoder_length のさらなる短縮
11. **min_encoder_length=10**: 15→10 でさらにサンプル増加。
12. **min_encoder_length=5**: 極端に短縮。

## 手順
1. ログ確認 → 悪化(ens_1d -0.015)→revert
2. 1つ実装（`# R8-XX:`）
3. 実行: `rm -f "Temporal Fusion Transformer/artifacts/feature_schema.json" && cd "Temporal Fusion Transformer" && uv run python train_tft.py 2>&1 | tail -30`
4. 確認: `uv run python -c "import json; m=json.load(open('Temporal Fusion Transformer/artifacts/optimization_log.json'))[-1]['metrics']; print(f'ens_1d={m[\"ensemble_direction_1d\"]}, ens_5d={m[\"ensemble_direction_5d\"]}')" `
5. 判断: +0.015→採用、-0.015→revert、横ばい→revert

## 制約
- train_tft.py のみ、新規パッケージ禁止、GPU/CPU両対応、ログ維持
- エラー時は即revert
- **NoisyDataLoader 等のカスタムクラスは train_tft.py 内に定義**

## 完了条件
`<promise>TFT OPTIMIZED</promise>`:
1. ens_1d >= 0.70 (達成済み: iter15 で 0.7225)
2. 8 iter 連続改善なし
3. 50 iter 到達
