# TFT USD/JPY 最適化ループ

## あなたの役割
Temporal Fusion Transformer/train_tft.py のFX予測性能を反復的に改善するAIエンジニア。
毎イテレーションで「1つの変更 → 実行 → 計測 → 判断」のサイクルを回す。

## 主要メトリクス（優先順）
1. direction_accuracy_5d（5日先方向精度）: 目標 >= 0.60
2. direction_accuracy（1日先方向精度）: 目標 >= 0.58
3. trade_sharpe_ratio: 目標 >= 1.0
4. mae: 最小化（現在のベースラインから改善）

## 毎イテレーションの手順

### Step 1: 状態確認
- Temporal Fusion Transformer/artifacts/optimization_log.json を読んで前回までの結果を確認
- 前回の変更が改善/悪化のどちらだったか判断
- 悪化した場合は前回の変更を revert してから次へ

### Step 2: 改善策を1つ選択
以下の候補から、まだ試していないものを1つ選ぶ（小さな変更に留める）:

ハイパーパラメータ調整:
- HIDDEN_SIZE (64, 128, 160, 256, 320)
- ATTENTION_HEAD_SIZE (1, 2, 4, 8)
- DROPOUT (0.05 ~ 0.4)
- HIDDEN_CONTINUOUS_SIZE (4, 8, 16, 32)
- LEARNING_RATE (5e-4 ~ 5e-3)
- MAX_ENCODER_LENGTH (30, 45, 60, 90, 120)
- MAX_PREDICTION_LENGTH (5, 10, 15, 20)
- DIRECTION_LOSS_WEIGHT (0.0, 0.5, 1.0, 2.0, 3.0)
- BATCH_SIZE (32, 64, 128)
- weight_decay in train_tft() (1e-3, 1e-2, 5e-2)

特徴量エンジニアリング:
- 新テクニカル指標追加（Stochastic, CCI, OBV, Williams%R, Ichimoku等）
- ラグ特徴量（close_lag_1, close_lag_5 等）
- ボラティリティ特徴量（realized volatility, Parkinson等）
- リターンの移動統計量（rolling skew, kurtosis）
- 出来高関連（yfinanceから取得可能なら）
- 既存の低重要度特徴量の除去（interpret()結果を参考に）

損失関数・学習:
- DirectionAwareQuantileLoss の改善
- quantiles の変更 ([0.05, 0.25, 0.5, 0.75, 0.95] 等)
- reduce_on_plateau_patience の調整
- gradient_clip_val の調整

前処理:
- 外れ値クリッピングの閾値変更（3σ, 5σ）
- 正規化手法の変更
- データ期間の変更（START_DATE調整）

アーキテクチャ:
- lstm_layers パラメータ追加
- output_size の調整

### Step 3: 変更を実装
- train_tft.py を編集（1つの変更のみ）
- 変更内容をコメントで記録

### Step 4: 実行・計測
feature_schema.json が存在する場合は削除してから実行:
  rm -f "Temporal Fusion Transformer/artifacts/feature_schema.json"
  cd "Temporal Fusion Transformer" && uv run python train_tft.py --quick

--quick モード: 1seed, 30epoch, patience=5 の高速評価

### Step 5: 結果記録
実行後の出力から以下を確認して報告:
- direction_accuracy, direction_accuracy_5d
- mae, rmse
- trade_sharpe_ratio, trade_win_rate, trade_profit_factor
- 前回比の改善/悪化

### Step 6: 判断
- 改善 → 変更を維持、次のイテレーションへ
- 悪化 → 次のイテレーション冒頭で revert
- 横ばい → 変更を維持するが、別の方向性を探る

## 制約
- train_tft.py 1ファイルのみ編集
- 既存の関数シグネチャは維持（引数追加はOK）
- 新しい pip パッケージは追加しない（ta, yfinance, pytorch-forecasting, lightning, optuna 等の既存パッケージのみ使用）
- GPU/CPU両対応を維持
- feature_schema.json が古くなった場合は削除して再生成させる

## 完了条件
以下のいずれかを満たしたら完了を宣言:
1. direction_accuracy_5d >= 0.60 かつ trade_sharpe_ratio >= 1.0
2. 5イテレーション連続で改善なし（収束と判断）
3. 50イテレーション到達
