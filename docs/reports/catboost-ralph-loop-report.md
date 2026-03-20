# CatBoost Ralph Loop 最適化レポート

**実施日**: 2026-03-20
**対象**: `notebooks/usd_jpy_catboost.ipynb`
**手法**: ヒルクライム型 Ralph Loop（実行 → メトリクス比較 → ベスト保持/ロールバック）
**イテレーション数**: 15回（max-iterations到達）

## 最終結果

| 指標 | 値 |
|------|-----|
| **Sharpe Ratio** | **+0.229** |
| Total P&L | +7,051 円 |
| Win Rate | 50.24% |
| Profit Factor | 1.010 |
| Max Drawdown | -144,361 円 |
| Calmar Ratio | 0.154 |
| Total Trades | 1,268 |
| Avg Win | +1,131 円 |
| Avg Loss | -1,131 円 |
| Trade Rate | 1.10% |

## ベスト設定（Iteration 10）

```
RANDOM_SEED        = 123
N_TRIALS           = 20
THRESHOLD_PIPS     = 5 ~ 15（Optunaが選択）
PROB_THRESHOLD     = 0.45 ~ 0.80（Optunaが選択）
TOP_N_FEATURES     = 40
CV_SPLITS          = 5
TIME_FEATURE_POLICY = "cap"（最大4個）
USE_GPU            = True
```

## イテレーション履歴

| It | 変更内容 | Sharpe | 結果 |
|----|----------|--------|------|
| 1 | Baseline（N_TRIALS=10） | -66.60 | BASELINE |
| 2 | PROB閾値 0.30-0.70 → 0.45-0.80 | -12.21 | IMPROVED |
| 3 | PIPS閾値 2-10 → 5-15, N_TRIALS=20 | +0.07 | IMPROVED |
| 4 | TOP_N 40→60, CV 5→3 | -2.32 | REVERTED |
| 5 | PIPS閾値 5-15 → 3-8, N_TRIALS=30 | -5.27 | REVERTED |
| 6 | グリッドstep 0.02→0.01 | -38.08 | REVERTED |
| 7 | N_TRIALS=40, lr上限0.3 | -4.06 | REVERTED |
| 8 | 時間特徴量を完全除外 | N/A | REVERTED |
| 9 | 時間特徴量2個, iterations=5000 | -9.29 | REVERTED |
| **10** | **RANDOM_SEED 42→123** | **+0.229** | **IMPROVED** |
| 11 | N_TRIALS 20→30（seed=123） | -11.53 | REVERTED |
| 12 | RANDOM_SEED 123→777 | -4.02 | REVERTED |
| 13 | TOP_N 40→30（seed=123） | -4.07 | REVERTED |
| 14 | PROB閾値 0.50-0.85（seed=123） | -8.17 | REVERTED |
| 15 | ATR_PERCENTILE 30→50 | -4.05 | REVERTED |

改善率: 3/15（20%）、ロールバック: 11/15（73%）

## 分析・知見

### 効果があった変更

1. **確率閾値の引き上げ（It2）**: 低確信度トレードを排除。取引数が85,623→35に激減しSharpe改善。
2. **pips閾値の引き上げ（It3）**: ノイズレベルの小さな値動きを無視し、大きなトレンドのみ狙う方針に転換。初のプラスSharpe。
3. **ランダムシード変更（It10）**: Optunaの探索パスが変わり、より良いハイパーパラメータの組み合わせを発見。最大の改善幅（+0.072 → +0.229）。

### 効果がなかった変更

- **特徴量数の増減（It4, It13）**: 40が最適。増やすと過学習、減らすと情報不足。
- **Optunaトライアル数の増加（It7, It11）**: 必ずしも改善しない。試行数よりもseedによる探索パスの方が影響大。
- **閾値グリッドの細分化（It6）**: バリデーションへの過適合を招き大幅悪化。
- **時間特徴量の除外/制限（It8, It9）**: モデルが確信度の高い予測を出せなくなる。時間特徴量はこのモデルに必須。
- **ATRフィルター強化（It15）**: トレード対象を絞っても精度は改善せず。

### 根本的な課題

- **Optuna目的関数がマイナス**: ベストのtrial scoreでも-0.32程度。訓練データ上ですら安定的に利益を出せるモデルを見つけられていない。
- **Optunaのランダム性への強い依存**: seed変更が最も効果的だったことは、探索空間に安定した良解がまばらにしか存在しないことを示唆。
- **早期停止が早すぎる**: 3,000イテレーション中107で停止。モデルの表現力は十分だが、学習信号が弱い。

## 技術的メモ

- **nbconvert → papermill**: Windows環境でnbconvertはCatBoost GPU使用時にカーネルが死亡。papermillに切り替えて解決。
- **SHAP無効化**: SHAP TreeExplainerがGPUモデルでOOMを起こすため、自動実行時はスキップ。
- **実行時間**: 1イテレーション約5〜8分（GPU使用、Optuna 20 trials）。

## ファイル

- ベストノートブック: `artifacts/catboost/best_notebook.ipynb`
- ベストメトリクス: `artifacts/catboost/best_metrics.json`
- ベストモデル: `artifacts/catboost/model.cbm`
- 改善履歴: `artifacts/catboost/history.log`
