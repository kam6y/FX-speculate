# Multi-Agent Prediction Model Design

## Overview

3つの独立したClaude Codeターミナルセッション（エージェント）が、それぞれゼロから自由にUSD/JPY 1分足の方向予測モデルを構築する。各エージェントは `/ralph-loop` で反復改善を行い、最終的に3つのモデルの性能を比較する。

## 設計方針

- **共有モジュール統一**: `scripts/data_loader.py`, `features.py`, `evaluation.py` は全エージェント共通で変更禁止
- **モデル自由**: アーキテクチャ、学習パイプライン、ハイパーパラメータ空間はエージェントが自由に選択
- **GPU活用**: 可能な限りGPUを使用（PyTorch CUDA, LightGBM GPU, CatBoost GPU等）
- **Notebookベース**: 既存notebookと同じ形式で `notebooks/agent_N.ipynb` を作成
- **成果物管理**: `artifacts/agent_N/` にベストモデルを上書き保存（サイクルごとの履歴は保持しない）

## ディレクトリ構成

```
FX-speculate/
├── notebooks/
│   ├── agent_1.ipynb              # エージェント1
│   ├── agent_2.ipynb              # エージェント2
│   ├── agent_3.ipynb              # エージェント3
│   ├── usd_jpy_transformer.ipynb  # 参考（既存）
│   ├── usd_jpy_lightgbm.ipynb     # 参考（既存）
│   └── usd_jpy_catboost.ipynb     # 参考（既存）
├── artifacts/
│   ├── agent_1/
│   │   ├── metrics.json
│   │   ├── config.pkl
│   │   └── model.*
│   ├── agent_2/
│   │   └── ...
│   └── agent_3/
│       └── ...
├── scripts/
│   ├── data_loader.py             # 共有（変更不可）
│   ├── features.py                # 共有（変更不可）
│   ├── evaluation.py              # 共有（変更不可）
│   └── compare_agents.py          # 新規: 結果比較
└── CLAUDE.md                      # エージェント共通ルール
```

## CLAUDE.md（エージェント共通ルール）

プロジェクトルートに配置。内容:

- 共有モジュールの変更禁止ルール
- 必須インポートパターン
- エージェントの自由範囲と制約
- metrics.json の必須フォーマット（compute_metrics()出力準拠）
- 各サイクルの流れ（モデル選択→Optuna→再学習→閾値最適化→バックテスト→保存）

## エージェントプロンプト

各ターミナルで `claude` 起動後に投入。エージェントIDのみ異なる共通テンプレート:

```
あなたはAgent Nです。notebooks/agent_N.ipynbを作成し、USD/JPY 1分足の方向予測モデルを
ゼロから構築してください。

CLAUDE.mdのルールに従うこと。既存のnotebooks/（transformer, lightgbm, catboost）を
参考にしつつ、自分で自由にモデルアーキテクチャを選んでください。
GPUを最大限活用すること。成果物はartifacts/agent_N/に保存。

notebookを作成したら実行し、バックテスト結果を確認してください。
完了後 /ralph-loop を開始し、各サイクルでモデルの改善を繰り返してください。
```

## 比較スクリプト（compare_agents.py）

`scripts/compare_agents.py` で3エージェントの `metrics.json` を横並び比較。
主要指標: sharpe_ratio, total_pnl, win_rate, profit_factor, max_drawdown, calmar_ratio, total_trades, trade_rate。
Sharpe Ratio基準でベストエージェントを表示。

## 反復改善サイクル

1. モデルアーキテクチャを選択または改善
2. Optunaでハイパーパラメータ最適化（trial数はエージェント裁量）
3. ベストパラメータで全訓練データで再学習
4. バリデーションセットで閾値最適化（Buy/Sell確率閾値）
5. テストセットでバックテスト実行
6. 前回のmetrics.jsonと比較し、改善時のみ上書き保存
7. 次サイクルで別アプローチまたはパラメータ空間を試す

## 運用手順

1. CLAUDE.md, compare_agents.py, agent用ディレクトリを作成
2. 3つのターミナルを開き、それぞれ `claude` を起動
3. 各ターミナルにエージェントプロンプトを投入
4. 各エージェントがnotebook作成→実行→`/ralph-loop`開始
5. 十分に回したら各ターミナルを停止
6. `python scripts/compare_agents.py` で結果比較
