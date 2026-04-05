# CLAUDE.md

## 禁止事項

- **RANDOM_SEED の変更は禁止**: シード変更による改善は本質的でない。モデル改善はアーキテクチャ、特徴量、損失関数等の構造的変更で行うこと。

## 開発メモ

### 実行方法
```bash
PYTHONPATH=. uv run python scripts/train.py    # 学習（2段階: Stage1 QuantileLoss → Stage2 方向FT）
PYTHONPATH=. uv run python scripts/evaluate.py  # 評価
PYTHONPATH=. uv run python -m pytest tests/ -v  # テスト
```

### 主要な設計判断
- **方向シグナル**: q50 単体ではなく全5分位点の加重平均 (upper_skew: [0.05, 0.10, 0.25, 0.30, 0.30]) で方向判定。これが最大の精度改善要因。
- **2段階学習**: Stage1 で基本予測力を習得、Stage2 で方向ペナルティによるファインチューニング。効果は +0.2pt と小さいが安定的。
- **閾値最適化**: `accuracy - 1.0 * ratio_gap` のハイブリッドスコアで tune セット上で 10000 グリッドで探索。ratio_gap ペナルティを強くすることでキャリブレーションと精度の両方が改善。
- **ドロップアウト**: 0.15 が最適。0.3 では小さいモデル（hidden_size=32）に対して過正則化。0.10 では過学習。
- **エンコーダ長**: 90営業日が最適。60→90でH2-H5の方向精度が改善。110以上はtuneセット（124サンプル）のウィンドウ数不足で閾値最適化が破綻。
- **学習率**: 7e-5 が encoder=90 での最適値。5e-5 から +0.3pt 改善。1e-4 では過大、3e-5 では収束が遅すぎ。
- **Stage 2 (方向FT)**: encoder=90 ではStage2無効化（direction_weight=0.0）が最良。Stage2はepoch 0で停止し実質的に機能しない。
- **方向ペナルティの温度**: SMOOTHING_TEMPERATURE=2.0 では log_return（~0.003）に対して tanh が線形領域になり、方向ペナルティが定数化して勾配に寄与しない。temp=0.005 で機能するが、dw=0.01 が最適で平均精度はdw=0.0と同等（H4/H5は改善するがH2が悪化）。
- **特徴量の安定性**: 現在の特徴量セットは最適化済み。追加（VIX term structure, cross-market lead-lag, COT）も削除（低重要度5特徴量）も方向精度を大幅に悪化させる。TFT の encoder が時系列パターンを自己学習するため、ラグ特徴量や移動平均の派生は冗長でノイズになる。
- **モデルサイズの安定性**: hidden_size=32 が最適。48, 64 では方向精度が50%以下に悪化。ATTENTION_HEAD_SIZE=4, HIDDEN_CONTINUOUS_SIZE=16 も同様に最適。
- **バッチサイズ**: 16 が最適。32→16 で平均方向精度 +0.8pt 改善（特にH1が+2.5pt）。小バッチのノイジーな勾配が暗黙の正則化として機能。64では悪化。gradient_clip_val（0.1/0.3/0.5）と reduce_on_plateau_patience（4/6/8）は結果に影響なし。
- **Weight decay は禁止**: Adam + weight_decay は 1e-4 でも方向精度が 50% 以下に崩壊（適応的学習率との干渉）。AdamW + weight_decay は無害だが改善もなし。現行の Adam, weight_decay=0 が最適。
