# Ralph Loop Prompt: CatBoost

以下の手順を毎イテレーション実行せよ。

## 設計書

`docs/superpowers/specs/2026-03-20-three-model-comparison-design.md` に従う。

## 対象ファイル

`notebooks/usd_jpy_catboost.ipynb` のみ変更可能。`scripts/` 配下の共通モジュールは絶対に変更するな。

## 手順

### Step 1: ベスト確認

`artifacts/catboost/best_metrics.json` が存在すれば読み、現在のベスト Sharpe Ratio を確認せよ。
`artifacts/catboost/history.log` があれば読み、過去の改善履歴を把握せよ。

### Step 2: 改善

ノートブックに **1つだけ** 改善を加えよ。改善の例:
- 特徴量選択数の変更（Top40 → Top50等）
- ハイパーパラメータ探索空間の調整
- Optunaトライアル数の変更
- 前処理の工夫
- 確率閾値グリッドサーチの調整
何を変更したか明確に記録せよ。

### Step 3: 実行

```bash
cd "C:/Users/daiya/OneDrive/ドキュメント/FX-speculate/notebooks" && uv run papermill usd_jpy_catboost.ipynb usd_jpy_catboost.ipynb --cwd .
```

エラーが出た場合はノートブックを修正して再実行せよ。

### Step 4: メトリクス確認

`artifacts/catboost/metrics.json` を読み、以下を確認:
- Sharpe Ratio
- Total P&L
- Win Rate
- Profit Factor
- Max Drawdown
- Total Trades

### Step 5: ベスト比較・更新

- `best_metrics.json` が存在しない（初回）→ 現在のメトリクスを `best_metrics.json` に保存、ノートブックを `artifacts/catboost/best_notebook.ipynb` にコピー
- Sharpe Ratio が改善 → `best_metrics.json` を更新、`best_notebook.ipynb` を上書き
- Sharpe Ratio が悪化 → `best_notebook.ipynb` から `notebooks/usd_jpy_catboost.ipynb` にリストア（ベスト版に戻す）

### Step 6: 履歴記録

`artifacts/catboost/history.log` に以下を追記:

```
--- Iteration N ---
Change: [何を変更したか]
Sharpe: [今回のSharpe] (Best: [ベストSharpe])
Result: [IMPROVED / REVERTED]
```

### 完了条件

Sharpe Ratio > 1.5 を達成し、かつ Total Trades > 100 の場合:

<promise>CATBOOST COMPLETE</promise>
