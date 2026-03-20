# Ralph Loop Prompt: Transformer

以下の手順を毎イテレーション実行せよ。

## 設計書

`docs/superpowers/specs/2026-03-20-three-model-comparison-design.md` に従う。

## 対象ファイル

`notebooks/usd_jpy_transformer.ipynb` のみ変更可能。`scripts/` 配下の共通モジュールは絶対に変更するな。

## 手順

### Step 1: ベスト確認

`artifacts/transformer/best_metrics.json` が存在すれば読み、現在のベスト Sharpe Ratio を確認せよ。
`artifacts/transformer/history.log` があれば読み、過去の改善履歴を把握せよ。

### Step 2: 改善

ノートブックに **1つだけ** 改善を加えよ。改善の例:
- ウィンドウサイズの探索範囲調整
- Transformerアーキテクチャの調整（d_model, nhead, num_layers等）
- 学習率やバッチサイズの探索範囲調整
- Optunaトライアル数の変更
- ドロップアウト率の調整
- 正規化手法の変更
何を変更したか明確に記録せよ。

### Step 3: 実行

```bash
cd "C:/Users/daiya/OneDrive/ドキュメント/FX-speculate/notebooks" && uv run papermill usd_jpy_transformer.ipynb usd_jpy_transformer.ipynb --cwd .
```

Transformerは学習に時間がかかるため、タイムアウトを長めに設定。
エラーが出た場合はノートブックを修正して再実行せよ。

### Step 4: メトリクス確認

`artifacts/transformer/metrics.json` を読み、以下を確認:
- Sharpe Ratio
- Total P&L
- Win Rate
- Profit Factor
- Max Drawdown
- Total Trades

### Step 5: ベスト比較・更新

- `best_metrics.json` が存在しない（初回）→ 現在のメトリクスを `best_metrics.json` に保存、ノートブックを `artifacts/transformer/best_notebook.ipynb` にコピー
- Sharpe Ratio が改善 → `best_metrics.json` を更新、`best_notebook.ipynb` を上書き
- Sharpe Ratio が悪化 → `best_notebook.ipynb` から `notebooks/usd_jpy_transformer.ipynb` にリストア（ベスト版に戻す）

### Step 6: 履歴記録

`artifacts/transformer/history.log` に以下を追記:

```
--- Iteration N ---
Change: [何を変更したか]
Sharpe: [今回のSharpe] (Best: [ベストSharpe])
Result: [IMPROVED / REVERTED]
```

### 完了条件

Sharpe Ratio > 1.5 を達成し、かつ Total Trades > 100 の場合:

<promise>TRANSFORMER COMPLETE</promise>
