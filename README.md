# TFT USD/JPY 予測AI

Google の [Temporal Fusion Transformer (TFT)](https://arxiv.org/abs/1912.09363) 論文に基づく、USD/JPY 為替レートの短期マルチホライズン予測システム。

## 概要

- **モデル:** pytorch-forecasting の TFT（分位点予測 + 方向ペナルティ付きカスタム損失）
- **予測:** 入力60営業日 → 5日先までのマルチホライズン予測
- **データ:** Yahoo Finance（為替・株価・商品） + FRED API（金利・マクロ経済指標）
- **出力:** 学習/評価スクリプト + Streamlit ダッシュボード + 日次自動予測

## セットアップ

### 前提条件

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) パッケージマネージャ
- NVIDIA GPU（推奨、CUDA 12.8 対応版 PyTorch を自動インストール）

### インストール

```bash
# 依存関係のインストール（dev ツール含む）
uv sync --extra dev

# .env に FRED API キーを設定
echo "FRED_API_KEY=your_api_key_here" > .env
```

FRED API キーは https://fred.stlouisfed.org/docs/api/api_key.html から無料で取得できる。

### GPU サポート

`pyproject.toml` で CUDA 12.8 対応版 PyTorch が設定済み。`uv sync` で自動的に GPU 版がインストールされる。

```bash
# GPU 認識の確認
uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## 使い方

### 1. 学習

```bash
# 通常学習（early stopping あり、最大100エポック）
uv run python scripts/train.py

# Optuna ハイパラチューニング（各トライアル30エポック）
uv run python scripts/train.py --optuna --n-trials 50
```

学習完了後、以下が生成される:
- `artifacts/checkpoints/` — val_loss 上位5件のモデルチェックポイント
- `artifacts/train_meta.json` — 学習メタデータ（データサイズ、ベスト val_loss 等）

### 2. 評価

```bash
uv run python scripts/evaluate.py
uv run python scripts/evaluate.py --top-k 5  # 上位5モデルでアンサンブル
```

評価完了後、以下が生成される:
- `artifacts/eval/eval_report.json` — ホライゾン別 MAE/RMSE/方向精度
- `artifacts/eval/direction_ratio.png` — 実績 vs 予測の方向比率グラフ
- `artifacts/thresholds.json` — ホライゾン別の最適方向閾値
- `artifacts/feature_importance.json` — TFT Variable Selection Network の特徴量重要度
- `artifacts/attention_weights.json` — Attention ヒートマップデータ

### 3. 日次予測

```bash
uv run python scripts/predict.py
```

最新データを取得し、top-k モデルのアンサンブルで5営業日先まで予測する。結果は `artifacts/predictions.db`（SQLite）に蓄積される。

### 4. ダッシュボード

```bash
uv run streamlit run dashboard/app.py
```

ブラウザで http://localhost:8501 にアクセス。8つのパネルで予測結果・モデル解釈・評価メトリクスを可視化する。

| パネル | 内容 |
|--------|------|
| 予測チャート | 5日先までのファンチャート（中央値 + 90% 信頼区間） |
| 方向シグナル | ホライゾン別 UP/DOWN 判定と閾値との差分 |
| 方向比率モニター | 実績 vs 予測の上昇比率比較（キャリブレーション確認） |
| イベントカレンダー | 直近30日の主要経済イベント一覧 |
| 特徴量重要度 | TFT Variable Selection Network による入力特徴量の寄与度 |
| Attention ヒートマップ | 各予測ホライゾンがどの過去時点に注目しているか |
| 過去予測の精度 | テストセットでの MAE/RMSE/方向精度テーブル + 折れ線グラフ |
| サイドバー | データ更新ボタン、予測履歴の統計 |

### 5. 定期実行（Windows タスクスケジューラ）

```bash
# 管理者権限で実行
scripts\schedule_task.bat
```

毎日 07:00 JST（NY市場クローズ後）に `predict.py` を自動実行する。

### 6. テスト

```bash
uv run pytest tests/ -v
```

## プロジェクト構成

```
FX-speculate/
├── config.py                  # 全設定の一元管理（パス・ハイパラ・定数）
├── pyproject.toml             # 依存関係・uv 設定（CUDA インデックス含む）
├── .env                       # 環境変数（FRED_API_KEY）
│
├── data/                      # データパイプライン
│   ├── fetch.py               # Yahoo Finance / FRED データ取得（公表ラグ適用済み）
│   ├── features.py            # 特徴量エンジニアリング（テクニカル・マクロ・カレンダー・イベント）
│   ├── events.py              # 経済イベントカレンダー（FOMC, BOJ, NFP, CPI 等）
│   └── dataset.py             # pytorch-forecasting TimeSeriesDataSet 構築
│
├── model/                     # モデル定義
│   ├── loss.py                # DirectionAwareQuantileLoss（tanh スムージング付き方向ペナルティ）
│   └── trainer.py             # Lightning Trainer + TFT モデル構築
│
├── scripts/                   # 実行エントリポイント
│   ├── train.py               # 学習（通常 + Optuna ハイパラチューニング）
│   ├── evaluate.py            # 評価・閾値チューニング・特徴量重要度/Attention 抽出
│   ├── predict.py             # 日次推論（encoder + decoder 構築 → SQLite 蓄積）
│   └── schedule_task.bat      # Windows タスクスケジューラ設定
│
├── dashboard/
│   └── app.py                 # Streamlit ダッシュボード（8パネル構成）
│
├── tests/                     # ユニット + 統合テスト
│   ├── test_fetch.py
│   ├── test_features.py
│   ├── test_events.py
│   ├── test_dataset.py
│   ├── test_loss.py
│   ├── test_trainer.py
│   └── test_integration.py
│
└── artifacts/                 # 出力（gitignore）
    ├── raw_data.parquet       # キャッシュ済み生データ（当日有効）
    ├── checkpoints/           # モデルチェックポイント（top-k）
    ├── eval/                  # 評価レポート・グラフ
    ├── predictions.db         # 予測結果蓄積 SQLite
    ├── train_meta.json        # 学習メタデータ
    ├── thresholds.json        # ホライゾン別方向閾値
    ├── feature_importance.json # VSN 特徴量重要度
    └── attention_weights.json  # Attention ヒートマップデータ
```

## パイプライン

```
Yahoo Finance + FRED API
        │
        ▼
   [fetch.py] ──→ raw_data.parquet（公表ラグ適用・ffill・キャッシュ）
        │
        ▼
 [features.py] ──→ テクニカル / リターン / マクロ / カレンダー / イベント特徴量
        │
        ▼
  [dataset.py] ──→ TimeSeriesDataSet（GroupNormalizer・encoder/decoder 分離）
        │
        ▼
   [train.py] ──→ TFT 学習（Lightning Trainer・top-3 チェックポイント保存）
        │
        ▼
 [evaluate.py] ──→ 閾値チューニング → テスト評価 → 特徴量重要度 / Attention 抽出
        │
        ▼
  [predict.py] ──→ 最新データ取得 → アンサンブル推論 → SQLite 蓄積
        │
        ▼
 [dashboard/app.py] ──→ Streamlit 可視化（8パネル）
```

## 特徴量

### 入力特徴量一覧

| カテゴリ | 特徴量 | TFT 分類 |
|----------|--------|----------|
| テクニカル | SMA(5,20,60), RSI(14), MACD, Bollinger Bands, ATR(14) | time-varying unknown |
| リターン | 対数リターン (1d, 5d, 20d) | time-varying unknown |
| 関連市場 | S&P500, 日経225, VIX, 原油, 金のリターン | time-varying unknown |
| マクロ | 日米10年金利, FF金利, CPI, 失業率, GDP, M2, DXY, 金利差 | time-varying unknown |
| カレンダー | 曜日, 月, 月末フラグ | time-varying known |
| イベント | 次の主要イベントまでの日数, 前回からの日数, 当日フラグ, 過去5日密度 | time-varying known |

### 公表ラグ

マクロ指標は公表日が参照期間の終端日より遅れる。`fetch.py` 段階で以下のラグを適用し、推論時と同じ条件でデータリーケージを防止している:

| 指標 | ラグ (営業日) |
|------|-------------|
| CPI | 35 |
| 失業率 | 32 |
| GDP | 30 |
| 日本10年金利 | 45 |
| M2 | 14 |
| FF金利 / 米10年金利 / DXY | 1 |

## モデルアーキテクチャ

### TFT 構成

| パラメータ | デフォルト値 |
|------------|-------------|
| エンコーダー長 | 60 営業日 |
| 予測長 | 5 営業日 |
| Hidden Size | 32 |
| Attention Head Size | 4 |
| Dropout | 0.3 |
| Hidden Continuous Size | 16 |
| 出力 | 5 分位点 (q10, q25, q50, q75, q90) |
| 学習率 | 1e-4 |

### DirectionAwareQuantileLoss

標準の QuantileLoss に方向ペナルティを追加したカスタム損失関数:

```
Loss = QuantileLoss + direction_penalty

方向ペナルティ:
  pred_dir  = tanh(pred_median / temperature)
  target_dir = tanh(target / temperature)
  mismatch  = (1 - pred_dir × target_dir) / 2
  penalty   = mismatch × direction_weight  (|target| < dead_zone のとき免除)
```

- `direction_weight = 1.0` — ペナルティの強さ
- `smoothing_temperature = 1.0` — tanh のスケール（GroupNormalizer 後の std ≈ 1 に合わせる）
- `dead_zone = 1e-4` — ほぼ無変動の日はペナルティを免除

中央値分位点 (q50) のみにペナルティを適用し、他の分位点の意味（不確実性の推定）を保持する。

### 学習設定

| 設定 | 値 |
|------|-----|
| 最大エポック | 100 |
| Early Stopping | val_loss 20エポック改善なしで停止 |
| バッチサイズ | 32 |
| Gradient Clipping | 0.1 |
| チェックポイント保存 | val_loss 上位 5 件 |
| ReduceOnPlateau | patience=4 |

### アンサンブル推論

top-k チェックポイント (デフォルト3) でアンサンブル推論を行い、単一モデルのばらつきを低減する:

- **中央値 (q50):** 各モデルの q50 の平均
- **下限 (q10):** 各モデルの q10 の最小値
- **上限 (q90):** 各モデルの q90 の最大値

## 方向比率キャリブレーション

モデルの予測 UP:DOWN 比率が実績に合うよう、3層で対策している:

1. **学習時:** `DirectionAwareQuantileLoss` の方向ペナルティで、方向の一致を促進
2. **評価時:** `direction_ratio_gap`（実績UP比率と予測UP比率の差）をメトリクスとして監視
3. **推論時:** threshold_tune セット (全データの5%) で、各ホライゾンごとに UP:DOWN 比率が実績に最も近くなる最適閾値を探索

### データ分割

```
|← train (65%) →|← val (15%) →|← tune (5%) →|← test (15%) →|
                                   ↑閾値チューニング専用
```

時系列順に固定分割し、未来のデータが学習に混入しないようにしている。

## Optuna ハイパラチューニング

`--optuna` フラグで以下のパラメータを自動探索する:

| パラメータ | 探索範囲 |
|------------|---------|
| hidden_size | {32, 64, 128} |
| dropout | 0.1 〜 0.4 (step=0.05) |
| learning_rate | 1e-4 〜 1e-2 (対数スケール) |
| direction_weight | 0.0 〜 3.0 (step=0.5) |

各トライアルは30エポックで打ち切り。結果は `artifacts/optuna_study.db` に保存され、`load_if_exists=True` で途中再開が可能。
