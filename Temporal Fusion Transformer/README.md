# TFT USD/JPY 予測AI

Temporal Fusion Transformer (TFT) による USD/JPY 為替レートの方向予測システム。
PyTorch Forecasting + PyTorch Lightning ベース。10日先までの多段階分位点予測とリアルタイムダッシュボードを提供する。

## ファイル構成

```
config.py        設定値・定数・パス・特徴量カテゴリ定義
model.py         DirectionAwareQuantileLoss / モデルロード / アンサンブル予測
data.py          データ取得・特徴量エンジニアリング・前処理・データセット作成
train.py         学習・評価・バックテスト・Walk-Forward・Optuna・デプロイ
evaluate.py      包括評価・9種の可視化グラフ生成
dashboard/
  app.py         FastAPI バックエンド + APScheduler 定時推論
  predictor.py   推論パイプライン・バックテストキャッシュ・実績更新
  database.py    SQLite スキーマ・CRUD (predictions / metrics / alerts)
  templates/     Jinja2 テンプレート
  static/        CSS (Kintsukuroi Terminal テーマ) + Chart.js フロントエンド
  model/         デプロイ済みチェックポイント (.ckpt)
artifacts/       学習成果物 (チェックポイント・メトリクス・評価レポート)
```

## セットアップ

```bash
# 依存パッケージ
uv add pytorch-forecasting lightning yfinance ta python-dotenv

# オプション
uv add fredapi       # FRED API (日米金利差)
uv add holidays      # 日米祝日フラグ

# ダッシュボード用
uv add fastapi uvicorn apscheduler jinja2
```

GPU がある場合は PyTorch の CUDA 版をインストールしておくこと。

## 使い方

### 学習

```bash
# 通常学習 (3 seed アンサンブル)
uv run python train.py

# Optuna ハイパラチューニング付き
uv run python train.py --optuna

# Walk-Forward バックテスト付き
uv run python train.py --walkforward

# 学習後にダッシュボードへモデルデプロイ
uv run python train.py --deploy
```

### 評価

```bash
# best 3 チェックポイントでアンサンブル評価
uv run python evaluate.py

# 特定のチェックポイントを指定
uv run python evaluate.py --ckpt path/to/model.ckpt

# top-5 アンサンブル
uv run python evaluate.py --top-k 5
```

`artifacts/eval/` に 9 種の PNG グラフと `eval_report.json` が出力される。

### ダッシュボード

```bash
cd "Temporal Fusion Transformer/dashboard"
uv run uvicorn app:app --reload --port 8501
```

`http://localhost:8501` でアクセス。17:30 JST (月〜金) に自動で翌営業日の予測を実行する。

## モデル概要

| 項目 | 値 |
|------|-----|
| アーキテクチャ | Temporal Fusion Transformer |
| 予測対象 | USD/JPY 対数リターン |
| 予測ホライズン | 1〜10 営業日 |
| 分位点 | q10 / q50 / q90 |
| エンコーダ長 | 120 日 |
| アンサンブル | 3 seed 平均 |
| 損失関数 | DirectionAwareQuantileLoss (方向ペナルティ付き) |

### 入力特徴量

- **Known categoricals**: day_of_week, month, quarter
- **Known reals**: 周期エンコーディング (dow/month/doy sin/cos), 月初・月末・四半期末フラグ, FOMC/BOJ/NFP 距離, 日米祝日
- **Unknown reals**: RSI(14), MACD diff, BB width, ATR(14), MA 乖離率 (5/20/50/200d), VIX, Gold, US10Y, 金利差, センチメント代理
- **Unknown categoricals**: market_regime (trend / range / high_vol)

## API エンドポイント

| エンドポイント | メソッド | 説明 |
|-------------|--------|------|
| `/api/predict` | GET | 明日の予測を実行 |
| `/api/predictions/latest` | GET | 最新の予測結果 |
| `/api/predictions/history` | GET | 直近 60 件の予測履歴 |
| `/api/backtest` | GET | バックテストデータ (キャッシュ) |
| `/api/backtest/refresh` | GET | バックテスト再計算 |
| `/api/metrics` | GET | モデルメトリクス |
| `/api/live-equity` | GET | DB 蓄積予測からのエクイティカーブ |
| `/api/update-actuals` | POST | 実績値を yfinance から更新 |
| `/api/alerts` | GET | アクティブアラート |
| `/api/alerts/{id}/resolve` | POST | アラート解決 |
| `/api/scheduler` | GET | スケジューラ状態 |
