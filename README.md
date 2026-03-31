# TFT USD/JPY 予測AI

Google の [Temporal Fusion Transformer (TFT)](https://arxiv.org/abs/1912.09363) 論文に基づく、USD/JPY 為替レートの短期マルチホライズン予測システム。

## 概要

- **モデル:** pytorch-forecasting の TFT（分位点予測 + 方向ペナルティ付きカスタム損失）
- **予測:** 入力60営業日 → 5日先までのマルチホライズン予測
- **データ:** Yahoo Finance（為替・株価・商品） + FRED API（金利・マクロ経済指標）
- **出力:** 学習/評価スクリプト + Streamlit ダッシュボード + 日次自動予測

## セットアップ

```bash
# uv がインストール済みであること
uv sync --extra dev

# .env に FRED API キーを設定
echo "FRED_API_KEY=your_api_key_here" > .env
```

## 使い方

### 学習

```bash
# 通常学習
uv run python scripts/train.py

# Optuna ハイパラチューニング
uv run python scripts/train.py --optuna --n-trials 50
```

### 評価

```bash
uv run python scripts/evaluate.py
uv run python scripts/evaluate.py --top-k 5
```

評価結果は `artifacts/eval/` に保存されます（`eval_report.json`, `direction_ratio.png`）。

### 日次予測

```bash
uv run python scripts/predict.py
```

予測結果は `artifacts/predictions.db`（SQLite）に蓄積されます。

### ダッシュボード

```bash
uv run streamlit run dashboard/app.py
```

### 定期実行（Windows タスクスケジューラ）

```bash
# 管理者権限で実行
scripts\schedule_task.bat
```

毎日 07:00 JST（NY市場クローズ後）に `predict.py` を自動実行します。

### テスト

```bash
uv run pytest tests/ -v
```

## プロジェクト構成

```
FX-speculate/
├── config.py              # 全設定の一元管理
├── data/
│   ├── fetch.py           # Yahoo Finance / FRED データ取得（公表ラグ適用済み）
│   ├── features.py        # テクニカル指標・リターン・マクロ・カレンダー・イベント特徴量
│   ├── events.py          # 経済イベントカレンダー（FOMC, NFP, CPI 等）
│   └── dataset.py         # pytorch-forecasting TimeSeriesDataSet 構築
├── model/
│   ├── loss.py            # DirectionAwareQuantileLoss（tanh スムージング付き）
│   └── trainer.py         # Lightning Trainer + TFT モデル構築
├── scripts/
│   ├── train.py           # 学習エントリポイント（通常 + Optuna）
│   ├── evaluate.py        # 評価・閾値チューニング・可視化
│   ├── predict.py         # 日次予測（encoder + decoder 構築 → SQLite 蓄積）
│   └── schedule_task.bat  # Windows タスクスケジューラ設定
├── dashboard/
│   └── app.py             # Streamlit ダッシュボード
├── tests/                 # 32 テスト
└── artifacts/             # チェックポイント・評価結果・予測ログ（gitignore）
```

## 特徴量

| カテゴリ | 特徴量 | TFT分類 |
|---|---|---|
| テクニカル | SMA(5,20,60), RSI, MACD, BB, ATR | time-varying unknown |
| リターン | 対数リターン(1d, 5d, 20d) | time-varying unknown |
| 関連市場 | S&P500, 日経225, VIX, 原油, 金 | time-varying unknown |
| マクロ | 日米金利差, CPI, 失業率, GDP, M2 | time-varying unknown |
| カレンダー | 曜日, 月, 月末フラグ | time-varying known |
| イベント | FOMC/日銀/NFP/CPI等までの日数, 密度 | time-varying known |

マクロ指標には公表ラグ（CPI: 35日, GDP: 30日等）を `fetch.py` 段階で適用し、データリーケージを防止しています。

## 方向比率キャリブレーション

モデルの予測 up:down 比率が実績に合うよう、3層で対策しています:

1. **学習時:** `DirectionAwareQuantileLoss` の方向ペナルティ
2. **評価時:** `direction_ratio_gap` メトリクス
3. **推論時:** 閾値チューニング専用セットで最適閾値を探索

データ分割: train(65%) : val(15%) : threshold_tune(5%) : test(15%)
