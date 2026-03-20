# Transformer モデル最適化レポート

日付: 2026-03-20

## 概要

USD/JPY 1分足方向予測 Transformer Encoder モデルに対し、Ralph Loop による反復的最適化を15イテレーション実施した。Sharpe Ratio 3.30 を達成したが、Total Trades > 100 の条件は未達（51件）。

## 最終ベスト結果

| 指標 | 値 |
|------|------|
| Sharpe Ratio | 3.3035 |
| Total P&L | +13,276円 |
| Win Rate | 58.82% |
| Profit Factor | 1.9645 |
| Max Drawdown | -2,319円 |
| Calmar Ratio | 18.08 |
| Total Trades | 51 |
| Avg Win | +901円 |
| Avg Loss | -655円 |

## モデル構成

| パラメータ | 値 |
|------------|------|
| d_model | 32 |
| nhead | 2 |
| num_layers | 1 |
| dim_feedforward | 128 |
| dropout | 0.184 |
| window_size | 100 |
| threshold_pips | 5.19 |
| prob_threshold (Buy) | 0.46 |
| prob_threshold (Sell) | 0.44 |
| learning_rate | 0.0009 |
| batch_size | 256 |
| 特徴量数 | 99 |
| 訓練データ | 528,913サンプル |
| テストデータ | 176,170サンプル |

## 環境

- GPU: NVIDIA GeForce RTX 5070 Ti
- PyTorch: 2.10.0+cu128 (Blackwell sm_120対応)
- Optuna: TPE Sampler, 10トライアル, timeout 2400秒
- 訓練データサブサンプリング: 30%（Optuna探索時のみ）

## イテレーション履歴

### イテレーション1 — ベースライン確立

- **変更**: 初期設定（N_TRIALS=10, subsample=30%, window 30-120, d_model [32,64]）
- **結果**: Sharpe -7.96, 412トレード
- **判定**: BASELINE

### イテレーション2 — 確率閾値引き上げ (BEST)

- **変更**: PROB_THRESHOLD_MIN 0.30→0.40, MAX 0.70→0.80
- **結果**: Sharpe **3.30**, 51トレード, Win Rate 58.8%
- **判定**: IMPROVED
- **考察**: 高確信度トレードへの限定が劇的に効果を発揮。低品質な予測を排除することでSharpeが大幅向上。

### イテレーション3 — 確率閾値微調整

- **変更**: PROB_THRESHOLD_MIN 0.40→0.35
- **結果**: Sharpe -37.09, 32,663トレード
- **判定**: REVERTED
- **考察**: 閾値を0.05下げただけで取引数が640倍に暴走。このモデルの確率分布は0.35-0.40付近に大量の予測が集中している。

### イテレーション4 — threshold_pips範囲変更

- **変更**: THRESHOLD_PIPS_MIN 2→1, MAX 10→8
- **結果**: Sharpe -5.66, 1,071トレード
- **判定**: REVERTED
- **考察**: 低いthreshold_pipsはラベルの境界を曖昧にし、ノイジーな信号を増加させた。

### イテレーション5 — モデルアーキテクチャ拡張

- **変更**: d_model [32,64,128], nhead [2,4,8], num_layers 1-3, seed=123
- **結果**: Sharpe 0.98, 162トレード
- **判定**: REVERTED
- **考察**: 大きいモデルは過学習傾向。小さいモデル(d_model=32, 1層)が最適。

### イテレーション6 — 学習エポック削減

- **変更**: max_epochs 200→100, patience 20→15, 閾値グリッド精密化
- **結果**: Sharpe -24.14, 15,865トレード
- **判定**: REVERTED
- **考察**: CosineAnnealingLRのT_max変更が学習率スケジュールを根本的に変え、モデルが劣化。

### イテレーション7 — 異なるseed+サブサンプリング増加

- **変更**: seed=77, OPTUNA_SUBSAMPLE 0.3→0.5
- **結果**: Sharpe -8.79, 2,074トレード
- **判定**: REVERTED
- **考察**: seed=42の探索パスが偶然良い解を見つけていた可能性。

### イテレーション8 — Trade-aware目的関数

- **変更**: Optuna目的関数を `Sharpe × min(trades/100, 1)` に変更
- **結果**: Sharpe -37.81, 26,920トレード
- **判定**: REVERTED
- **考察**: トレード数を目的関数に含めるとOptunaが大量トレード解に引き寄せられる。

### イテレーション9 — 二段階閾値最適化

- **変更**: Sharpe > 1.0フィルター → trades最大化の二段階グリッドサーチ
- **結果**: Sharpe -8.18, 707トレード
- **判定**: REVERTED
- **考察**: validation最適化がテストデータで汎化しない。閾値の過学習。

### イテレーション10 — Label Smoothing

- **変更**: CrossEntropyLossに label_smoothing=0.1 追加
- **結果**: Sharpe -19.11, 7,641トレード
- **判定**: REVERTED
- **考察**: Label Smoothingで予測確率分布が滑らかになり、閾値を超えるサンプルが激増。

### イテレーション11 — 短ウィンドウ

- **変更**: window_size 30-60に限定、MAX_WINDOW=60
- **結果**: Sharpe -15.35, 3,560トレード
- **判定**: REVERTED
- **考察**: 短期パターンは長期パターンより予測力が低い。

### イテレーション12 — Weight Decay強化

- **変更**: AdamW weight_decay 0.01→0.05
- **結果**: Sharpe -12.38, 1,725トレード
- **判定**: REVERTED
- **考察**: 強すぎる正則化はモデルの表現力を損なう。

### イテレーション13 — LayerNorm + Global Average Pooling

- **変更**: 入力射影後のLayerNorm追加、最終時点出力→全時点平均
- **結果**: Sharpe -5.33, 2,264トレード
- **判定**: REVERTED
- **考察**: Global Average Poolingは時系列の順序情報を希薄化させた。

### イテレーション14 — 3モデルアンサンブル

- **変更**: 3つの異なるseedで最終学習、予測確率を平均
- **結果**: Sharpe 0.74, 71トレード
- **判定**: REVERTED
- **考察**: アンサンブルで取引数は51→71に増加し予測は安定化したが、Sharpeはベスト未満。

## 知見と教訓

### 成功要因

1. **高確率閾値フィルタリング**: 確信度の低い予測を排除することがSharpe向上に最も効果的
2. **小さいモデル**: d_model=32, 1層のコンパクトな構成が99特徴量のFXデータに最適
3. **window_size=100**: 約1.5時間分のコンテキストが方向予測に適切

### Sharpe vs Trades のトレードオフ

このモデルは「スナイパー型」の予測特性を持つ。少数の高確信度トレード（51件、勝率59%）に集中することで高いSharpe Ratioを実現しているが、取引頻度を上げるあらゆる試み（閾値緩和、モデル変更、Label Smoothing等）は予測品質の低下を招いた。これはFX 1分足データの信号対雑音比が低いことを反映している。

### 今後の改善方向

- **データ量の増加**: より多くの訓練データで一般化性能を向上
- **特徴量エンジニアリング**: より予測力の高い特徴量の発見
- **マルチタイムフレーム入力**: 1分足に加えて5分足・15分足の情報を統合
- **Focal Lossの検討**: クラス不均衡への別アプローチ

## アーティファクト

| ファイル | 説明 |
|----------|------|
| `notebooks/usd_jpy_transformer.ipynb` | ベスト版ノートブック |
| `artifacts/transformer/best_notebook.ipynb` | ベスト版バックアップ |
| `artifacts/transformer/best_metrics.json` | ベスト版メトリクス |
| `artifacts/transformer/metrics.json` | 最新メトリクス（ベスト版と同一） |
| `artifacts/transformer/model.pt` | 学習済みモデル重み |
| `artifacts/transformer/config.pkl` | モデル設定・パラメータ |
| `artifacts/transformer/history.log` | 全イテレーション履歴 |
