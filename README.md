# FX-Speculate

USD/JPY の1分足データを用いた FX 方向予測モデル。CatBoost ベースの3クラス分類（Buy / Sell / Hold）で、15分先の値動きを予測し、バックテストで収益性を検証する。

## フォルダ構成

```
FX-speculate/
├── data/                       # 学習・評価データ
│   └── usd_jpy_1min_*.parquet
├── notebooks/                  # Jupyter ノートブック
│   ├── usd_jpy_model_v5.ipynb
│   └── usd_jpy_model_v6.ipynb
├── artifacts/                  # モデル出力 (gitignore, 実行時に自動生成)
├── pyproject.toml
├── uv.lock
└── README.md
```

## アーキテクチャ

```
1分足データ (Parquet)
  → TA-Lib 全指標 + 独自特徴量（MA乖離率, リターン, 時間特徴量 etc.）
    → Purged CV で特徴量選択 (Top 40)
      → Optuna でハイパーパラメータ + pips閾値 + 確率閾値を同時最適化
        → CatBoost 3クラス分類 (Buy / Hold / Sell)
          → ATR + 時間帯フィルタ付きバックテスト
```

### 主な設計判断

- **Purged Time Series Split**: 時系列データのリーク防止のため、train/valの間にgap（= 予測期間）を設ける
- **Trading Objective**: 分類精度ではなく Sharpe Ratio を直接最適化
- **コスト込み評価**: スプレッド (0.4 pips) + スリッページ (0.05 pips) をバックテストに織り込み
- **時間特徴量制御**: 過学習リスクの高い時間特徴量の採用数を cap で制限（v6〜）

## ノートブック

| ファイル | 概要 |
|---|---|
| `notebooks/usd_jpy_model_v5.ipynb` | 基本版。Purged CV + Trading Sharpe 最適化 |
| `notebooks/usd_jpy_model_v6.ipynb` | v5 改良版。時間特徴量 cap、Bid/Ask 対応バックテスト、ライブフィルタ統一 |

各ノートブックは上から順にセルを実行すると、データ読み込み → 特徴量生成 → モデル学習 → 評価 → リアルタイム推論まで一貫して動作する。

## データ

- `data/usd_jpy_1min_20231028_20260131_utc.parquet` — USD/JPY 1分足（約83万行）
- 学習・評価に必要。gitignore 対象のため、別途配置が必要

## リアルタイム推論

各ノートブックの Cell 10 で GMO コイン外国為替 FX Public API から直近5日分の1分足を取得し、学習済みモデルで売買シグナルを出力する。

フィルタ条件：
- UTC 20〜23時（低流動性時間帯）はスキップ
- ATR が訓練データの30パーセンタイル未満はスキップ

## セットアップ

### 1. uv をインストール（初回のみ）

PowerShell:
```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

### 2. Python + 依存パッケージをインストール

```bash
uv python install 3.11
uv sync
```

### 3. JupyterLab を起動

```bash
uv run jupyter lab
```

### GPU について

ノートブックのデフォルトは `USE_GPU = True`。CUDA が利用できない場合は Cell 0 で `USE_GPU = False` に変更する。

## 技術スタック

- **Python 3.11** / **uv** (パッケージ管理)
- **CatBoost** / LightGBM / XGBoost（モデル）
- **Optuna**（ハイパーパラメータ最適化）
- **SHAP**（特徴量解釈）
- **pandas** / **NumPy** / **ta**（データ処理・テクニカル指標）
- **Matplotlib** / **Seaborn**（可視化）
