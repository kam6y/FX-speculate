# TFT Horizon Accuracy 改善設計書

## 概要

1日方向精度(72.25%)を維持しつつ、全Horizon(1d-10d)の方向精度を底上げする。
現在のTFT + PyTorch Forecastingの枠内で、Loss関数の改良と特徴量追加の2軸で改善する。

## 背景・課題

### 現状パフォーマンス
| メトリクス | 値 |
|-----------|-----|
| Ensemble 1d方向精度 | 72.25% |
| Ensemble 5d方向精度 | 38.44% |
| Sharpe Ratio | 2.9 |
| Profit Factor | 1.62 |

### 根本原因: 評価と学習の不整合
- **評価**: 累積リターンベースで方向判定 (`actuals[:, :h_end].sum(dim=1)`)
- **学習**: 各ステップ独立で方向ペナルティ計算
- 評価では「5日間の合計がプラスか」を見るが、学習では「5日目単体の符号」しかペナルティしていない
- この不整合が中期Horizon精度の低迷の主因

## 設計

### 変更1: 累積方向ペナルティ Loss

`DirectionAwareQuantileLoss` を改良し、累積リターンベースの方向ペナルティを追加する。

#### 損失関数の構造

```
Total Loss = quantile_loss
           + α × per_step_direction_penalty   (既存)
           + β × cumulative_direction_penalty  (新規)
```

#### 累積方向ペナルティの計算

1. 予測median と target それぞれの cumsum を計算
2. 各Horizon h ∈ [1, 2, 3, ..., 10] について:
   - `cum_pred_h = pred_median[:, :h].sum(dim=1)`
   - `cum_actual_h = target[:, :h].sum(dim=1)`
3. 符号不一致の場合: `penalty_h = |cum_pred_h - cum_actual_h|`
4. 全Horizonの平均をペナルティとする

#### パラメータ

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| α (DIRECTION_LOSS_WEIGHT) | 0.3 | 個別ステップ方向重み (0.5→0.3に変更) |
| β (CUMULATIVE_DIRECTION_WEIGHT) | 0.3 | 累積方向重み (新規) |
| 累積ペナルティ対象Horizon | [1-10] | 全Horizon |

- 1d の実効方向重み = α + β = 0.6 (現行0.5と同程度で1d精度を維持)
- αを0.5→0.3に下げることで、h=1の重複によるバイアスを相殺

#### 実装箇所
- `train_tft.py`: `DirectionAwareQuantileLoss.loss()` メソッドを拡張
- `train_tft.py`: CONFIG に `CUMULATIVE_DIRECTION_WEIGHT: 0.3` を追加
- `train_tft.py`: CONFIG の `DIRECTION_LOSS_WEIGHT` を `0.5 → 0.3` に変更

### 変更2: マルチスケール特徴量追加

日次指標に加え、中期スケールの特徴量を追加してモデルに中期トレンド情報を与える。

#### 追加特徴量

| 特徴量 | 計算方法 | 目的 |
|--------|---------|------|
| `return_5d` | 5日間の累積log_return (`rolling(5).sum()`) | 週次モメンタム |
| `return_10d` | 10日間の累積log_return (`rolling(10).sum()`) | 2週間モメンタム |
| `rsi_5` | RSI(5) | 短期過熱感 |
| `volatility_10d` | 10日間log_returnの標準偏差 (`rolling(10).std()`) | 中期ボラティリティ |

#### 実装箇所
- `train_tft.py`: `add_technical_indicators()` 内に4つの特徴量計算を追加
- `train_tft.py`: `unknown_reals` リストに4つ追加
- NaN処理は既存の前処理フロー (ffill + dropna) で対応

### 変更しないもの
- モデルアーキテクチャ (hidden_size=192, attention_head_size=2, dropout=0.15 等)
- データ分割比率 (train:70%, val:15%, test:15%)
- 前処理パイプライン (外れ値クリッピング、正規化等)
- 評価コード (`evaluate_tft.py`)
- ダッシュボード

## リスクと対策

| リスク | 対策 |
|--------|------|
| 1d精度の低下 | α+β=0.6で現行0.5と同程度を維持。低下時はα微調整 |
| 学習不安定化 | 累積ペナルティは全Horizon平均で勾配を平滑化。gradient_clip_val=1.0維持 |
| 特徴量追加による過学習 | dropout=0.15、early_stopping patience=8 で制御。VSNが自動で重み調整 |

## 成功基準

- 1d方向精度: 70%以上を維持 (現行72.25%)
- 5d方向精度: 45%以上 (現行38.44%から+6.5pt以上)
- 全Horizon (2d-10d) で現行比改善
