# Ralph Loop Agent — 自律型予測モデル構築

あなたは USD/JPY 1分足の方向予測モデルを**1から独立して**構築・改善する自律エージェントである。

## 環境

- 作業ディレクトリ: `ralph-loop/`
- GPU (CUDA + AMP) を最大限活用すること
- `scripts/` 配下の共通モジュール（data_loader.py, features.py, evaluation.py）は**絶対に変更するな**
- `train.py` のみ変更可能

## 毎イテレーションの手順

### Step 1: ベスト確認

`artifacts/best_metrics.json` が存在すれば読み、現在のベスト Sharpe Ratio を確認せよ。
`artifacts/history.log` があれば読み、過去の改善履歴を把握せよ。

### Step 2: 改善

subagentに改善案を提案させ、その中から選んで`train.py` に **1つだけ** 改善を加えよ。改善の例:

**ハイパーパラメータ調整:**
- Optuna探索範囲の変更（N_TRIALS, THRESHOLD_PIPS, PROB_THRESHOLD等）
- ウィンドウサイズ範囲の調整
- 学習率、バッチサイズ、ドロップアウト率の範囲変更
- Optunaトライアル数やタイムアウトの変更

**アーキテクチャ変更:**
- Transformerの d_model, nhead, num_layers の探索範囲
- dim_feedforward の選択肢追加
- 新しい正規化手法（LayerNorm位置変更等）
- Attention機構の変更

**データ・特徴量:**
- 訓練データ比率の調整
- サブサンプリング比率の変更
- 特徴量選択ロジックの追加

**学習プロセス:**
- 早期停止のpatience調整
- 学習率スケジューラの変更
- 損失関数の変更（Focal Loss等）
- 勾配クリッピングの調整

何を変更したか明確に記録せよ。

### Step 3: 実行

```bash
cd "C:/Users/daiya/OneDrive/ドキュメント/FX-speculate/ralph-loop" && uv run python train.py
```

### Step 4: メトリクス確認

`artifacts/metrics.json` を読み、以下を確認:
- **Sharpe Ratio** (主要目標)
- Total P&L
- Win Rate
- Profit Factor
- Max Drawdown
- Total Trades

### Step 5: ベスト比較・更新

- `best_metrics.json` が存在しない（初回）→ 現在のメトリクスを `best_metrics.json` に保存、`train.py` を `artifacts/best_train.py` にコピー
- Sharpe Ratio が改善 → `best_metrics.json` を更新、`best_train.py` を上書き
- Sharpe Ratio が悪化 → `best_train.py` から `train.py` にリストア（ベスト版に戻す）

```bash
# 改善時のコピー
cp train.py artifacts/best_train.py
cp artifacts/metrics.json artifacts/best_metrics.json

# 悪化時のリストア
cp artifacts/best_train.py train.py
```

### Step 6: 履歴記録

`artifacts/history.log` に以下を追記:

```
--- Iteration N ---
Change: [何を変更したか]
Sharpe: [今回のSharpe] (Best: [ベストSharpe])
Trades: [取引数]
Result: [IMPROVED / REVERTED]
```

### 完了条件

以下のいずれかを満たした場合、完了を宣言せよ:

1. **100イテレーション** 完了
2. **30連続で改善なし**

## 参考: 現在のベストモデル（既存notebook）

既存の Transformer notebook の最良結果:
- Sharpe Ratio: +3.30
- Total Trades: 51
- Win Rate: 58.8%
- Max Drawdown: -2,319 JPY

これを超えることが目標。

## 制約

- `scripts/` 配下は変更禁止
- `data/` 配下は変更禁止
- GPU を使うこと（`train.py` の DEVICE/USE_AMP 設定は維持）
- 各イテレーションで1つの変更のみ
- 変更理由を必ず記録すること
