# LightGBM 最適化レポート

- 実施日: 2026-03-20
- 対象: `notebooks/usd_jpy_lightgbm.ipynb`
- 手法: Ralph Loop による反復改善（最大15イテレーション）
- 目標: Sharpe Ratio > 1.5 かつ Total Trades > 100

## 最終結果

| 指標 | 値 |
|------|-----|
| Sharpe Ratio | **-0.968** |
| Total P&L | -16,890 円 |
| Win Rate | 45.0% |
| Profit Factor | 0.904 |
| Max Drawdown | -74,619 円 |
| Calmar Ratio | 0.715 |
| Total Trades | 200 |
| Avg Win | +1,767 円 |
| Avg Loss | -1,599 円 |

**目標未達**。Sharpe Ratio > 1.5 は達成できなかった。

## ベスト構成

```python
CONFIG = {
    "THRESHOLD_PIPS_MIN": 2,
    "THRESHOLD_PIPS_MAX": 10,
    "THRESHOLD_PIPS_DEFAULT": 5,
    "TOP_N_FEATURES": 40,
    "RANDOM_SEED": 42,
    "N_TRIALS": 15,
    "CV_SPLITS": 3,
    "PROB_THRESHOLD_MIN": 0.45,
    "PROB_THRESHOLD_MAX": 0.75,
    "TIME_FEATURE_POLICY": "cap",
    "MAX_TIME_FEATURES": 4,
}

# Optunaハイパーパラメータ探索空間
objective_params = {
    "n_estimators": 500,          # 1000→500に削減
    "learning_rate": (0.01, 0.05),  # log scale
    "num_leaves": (15, 63),
    "max_depth": (3, 7),
    "min_child_samples": (20, 200),
    "reg_alpha": (0.1, 50.0),       # log scale
    "reg_lambda": (0.1, 50.0),      # log scale
    "subsample": (0.5, 0.9),
    "colsample_bytree": (0.4, 0.8),
}
early_stopping_rounds = 20  # CV内

# 最終モデル学習
final_n_estimators = 3000
final_early_stopping = 150
```

## イテレーション履歴

| # | 変更内容 | Sharpe | 結果 |
|---|---------|--------|------|
| 1 | ベースライン (N_TRIALS=15, CV=3) | -6.098 | BASELINE |
| 2 | 確率閾値引き上げ (0.30-0.70 → 0.45-0.75) | -2.421 | IMPROVED |
| 3 | 特徴量数 40→55 | -3.052 | REVERTED |
| 4 | threshold_pips範囲 2-10→3-8 | -5.490 | REVERTED |
| 5 | N_TRIALS 15→25 | -5.716 | REVERTED |
| 6 | 正則化強化 (lr, leaves, reg等) | -2.311 | IMPROVED |
| 7 | 時間特徴量完全除外 | -3.410 | REVERTED |
| 8 | グリッドサーチ細粒化 (0.02→0.01) | -4.468 | REVERTED |
| 9 | 確率閾値さらに引き上げ (0.50-0.80) | ERROR | REVERTED |
| 10 | Top30特徴量 + seed変更 | -6.502 | REVERTED |
| 11 | CV内 n_estimators=500, early_stop=20 | **-0.968** | IMPROVED |
| 12 | 最終モデル n_est=1500, early=50 | -1.135 | REVERTED |
| 13 | 確率閾値引き下げ (0.40-0.70) | -4.703 | REVERTED |
| 14 | CV内 n_est=200, early=10 | -1.812 | REVERTED |
| 15 | MAX_TIME_FEATURES 4→2 | -6.363 | REVERTED |

改善率: 3/15 (20%)

## 改善に寄与した変更

### 1. 確率閾値の引き上げ (イテレーション 2)

- PROB_THRESHOLD_MIN: 0.30 → 0.45
- PROB_THRESHOLD_MAX: 0.70 → 0.75
- 効果: Sharpe -6.098 → -2.421 (改善幅 3.677)
- 理由: 低確信度のノイズトレードを排除。トレード数が1,869→879に半減し、質が向上。

### 2. ハイパーパラメータ探索空間の正則化強化 (イテレーション 6)

- learning_rate: 0.005-0.12 → 0.01-0.05 (上限を大幅に下げ)
- num_leaves: 15-127 → 15-63 (複雑なツリーを制限)
- max_depth: 4-10 → 3-7 (浅いツリーに制限)
- min_child_samples: 5-100 → 20-200 (葉ノードの最小サンプル数増加)
- reg_alpha/lambda: 1e-8-10 → 0.1-50 (正則化の下限を大幅に引き上げ)
- 効果: Sharpe -2.421 → -2.311 (改善幅 0.110)
- 理由: 過学習抑制により汎化性能が向上。

### 3. CV内のモデル複雑度削減 (イテレーション 11)

- n_estimators: 1000 → 500 (ブースティングラウンド半減)
- early_stopping: 50 → 20 (より早期に停止)
- 効果: Sharpe -2.311 → -0.968 (改善幅 1.343)
- 理由: CV内での過学習を大幅に抑制。Optunaがよりロバストなパラメータ組み合わせを発見。avg_win (1,767) > avg_loss (1,599) に逆転。

## 悪化した変更からの教訓

| 変更 | 教訓 |
|------|------|
| 特徴量数の増加 (40→55) | ノイズ特徴量が増え予測精度が低下 |
| threshold_pips範囲の縮小 | 探索空間の制限はOptuna効率を下げる |
| N_TRIALSの増加 (15→25) | ランダムサンプリングの探索回数増加は限定的効果 |
| 時間特徴量の除外 | hour_sin/cos等は有益な情報を含む |
| グリッドサーチの細粒化 | val setへの過学習を招く |
| 確率閾値のさらなる引き上げ | トレード数がゼロになるリスク |
| CV n_est=200は過剰削減 | モデルの表現力が不足しトレード数激減 |

## 考察

### 根本的な課題

1分足USD/JPY方向予測はノイズ比率が極めて高く、取引コスト（スプレッド0.2pips + スリッページ0.1pips + API手数料0.002%）を超えるエッジを安定して確保することが困難。ベストモデルでもSharpe=-0.968と負の値であり、コスト控除後に利益を出せていない。

### 改善の方向性

今後の改善候補（優先度順）:

1. **データ量の増加**: より長期間のデータで学習し、レジーム変化に対するロバスト性を向上
2. **マルチタイムフレーム特徴量**: 5分・15分・1時間足のテクニカル指標を追加
3. **アンサンブル**: 複数seedで学習したモデルの予測を平均化
4. **予測ホライゾンの変更**: 15分→30分や60分に拡大し、シグナル/ノイズ比を改善
5. **ターゲット定義の見直し**: pips閾値ベースではなくボラティリティ調整済みターゲット

## アーティファクト

- ベストノートブック: `artifacts/lightgbm/best_notebook.ipynb`
- ベストメトリクス: `artifacts/lightgbm/best_metrics.json`
- 改善履歴: `artifacts/lightgbm/history.log`
- モデル: `artifacts/lightgbm/model.txt`
- 選択特徴量: `artifacts/lightgbm/selected_features.pkl`
- 設定: `artifacts/lightgbm/config.pkl`
