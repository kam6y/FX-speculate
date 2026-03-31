# TFT USD/JPY 予測AI 設計書

## 概要

Google の Temporal Fusion Transformer (TFT) 論文に基づき、USD/JPY 為替レートの短期マルチホライズン予測を行う AI システム。
`pytorch-forecasting` ライブラリを活用し、シンプルかつ保守しやすい設計を最優先とする。

## 要件

- **モデル:** TFT (pytorch-forecasting)
- **ターゲット:** USD/JPY 対数リターンの 1d〜5d 先マルチホライズン予測
- **入力ウィンドウ:** 60営業日
- **予測ホライズン:** 5営業日
- **データソース:** Yahoo Finance API + FRED API
- **出力:** 学習/評価スクリプト + Streamlit ダッシュボード + 日次自動予測

## プロジェクト構造

```
FX-speculate/
├── .env                          # FRED_API_KEY
├── pyproject.toml                # uv管理、依存関係
├── config.py                     # 全設定を一元管理
├── data/
│   ├── fetch.py                  # Yahoo Finance / FRED からの生データ取得
│   ├── features.py               # テクニカル指標・特徴量エンジニアリング
│   ├── events.py                 # 経済イベントカレンダー（FOMC, 雇用統計等）
│   └── dataset.py                # pytorch-forecasting用 TimeSeriesDataSet 構築
├── model/
│   ├── loss.py                   # カスタム損失関数
│   └── trainer.py                # 学習ループ（Lightning Trainer設定）
├── scripts/
│   ├── train.py                  # 学習実行エントリポイント
│   ├── evaluate.py               # 評価・可視化
│   ├── predict.py                # 日次予測（定期実行用）
│   └── schedule_task.bat         # Windowsタスクスケジューラ設定用
├── dashboard/
│   └── app.py                    # Streamlitダッシュボード
└── artifacts/                    # チェックポイント、評価結果、予測ログ（gitignore）
```

**設計方針:**
- 1ファイル1責務。モノリシックなスクリプトを避ける
- `config.py` で全パラメータを一元管理し、各モジュールはそこから import
- `data/` → `model/` → `scripts/` の依存方向を一方向に保つ
- `artifacts/` は gitignore 対象

## データ取得 (`data/fetch.py`)

### データソース

| ソース | データ | ライブラリ |
|---|---|---|
| Yahoo Finance | USD/JPY (OHLCV), S&P500, 日経225, VIX, 原油(WTI), 金(GC) | `yfinance` |
| FRED API | 米10年金利, 日10年金利, FF金利, CPI, 失業率, GDP, M2マネーサプライ, ドルインデックス(DXY) | `fredapi` |

- 取得期間: デフォルト10年分
- FREDの低頻度データ（CPI=月次、GDP=四半期）は前方補間(forward fill)で日次に揃える
- キャッシュ: 取得した生データを `artifacts/raw_data.parquet` に保存し、当日中は再取得しない
- FX営業日の取り扱い: yfinance から取得した USD/JPY の取引日インデックスをマスターカレンダーとして使用。FRED データとのマージ時は `pd.merge_asof` で最近の公表値を結合し、日付欠損を防ぐ

## 特徴量エンジニアリング (`data/features.py`)

| カテゴリ | 特徴量 | TFT分類 |
|---|---|---|
| テクニカル | SMA(5,20,60), RSI(14), MACD, ボリンジャーバンド, ATR | time-varying observed |
| リターン | 対数リターン(1d, 5d, 20d) | time-varying observed |
| 関連市場 | S&P500, 日経225, VIX, 原油, 金の各リターン | time-varying observed |
| マクロ | 日米金利差, CPI, 失業率, GDP, M2 | time-varying unknown |
| カレンダー | 曜日, 月, 月末フラグ | time-varying known |
| イベント | 後述の経済イベント特徴量 | time-varying known |

**マクロ特徴量の公表ラグ処理:**
CPI（翌月中旬）、失業率（翌月第1金曜）、GDP（四半期末+約1ヶ月）等は公表に遅延がある。
FRED API から取得した値をそのまま forward fill するのではなく、**実際の公表日ベースでラグを適用**した上で forward fill する。
これにより予測時点で未公表の値がモデルに漏れることを防ぐ。
分類は `time_varying_unknown_reals`（過去の値のみ参照可能）とする。

**公表ラグの実装方法:**
- `config.py` に各指標の標準公表ラグ日数を定義（例: `{"CPI": 35, "GDP": 30, "UNEMPLOYMENT": 32}`）
- `data/features.py` の共通関数 `apply_publication_lag(series, lag_days)` でラグを適用
- 学習時・推論時の両方で同じ関数を呼び出すことで、ラグ処理の一貫性を保証
- 推論時は「当日時点で公表済みの最新値」のみが入力されることを保証する

### ターゲット変数

単一ターゲット `log_return`（翌日の対数リターン）を設定し、`max_prediction_length=5` で5ステップ先（1d〜5d）を自動予測する。
TFT論文の本来のマルチホライズン設計に準拠し、`pytorch-forecasting` の正規パスを使用する。

```python
TimeSeriesDataSet(
    data,
    target="log_return",            # 単一ターゲット
    max_prediction_length=5,         # 5ホライズン分を自動予測
    max_encoder_length=60,
    time_varying_unknown_reals=["log_return", ...],
    ...
)
```

## 経済イベントカレンダー (`data/events.py`)

### 対象イベント

| イベント | 頻度 | 日程の特徴 |
|---|---|---|
| FOMC | 年8回 | 不定期、事前公表 |
| 日銀金融政策決定会合 | 年8回 | 不定期、事前公表 |
| 米雇用統計(NFP) | 月次 | 原則第1金曜日 |
| 米CPI | 月次 | 中旬、事前公表 |
| 米GDP速報 | 四半期 | 月末、事前公表 |
| 米ISM製造業景況指数 | 月次 | 第1営業日 |
| 米小売売上高 | 月次 | 中旬 |
| ジャクソンホール会議 | 年1回 | 8月下旬 |

### イベント特徴量

| 特徴量 | 説明 |
|---|---|
| `days_to_next_major_event` | 次の重要イベントまでの営業日数 |
| `days_from_last_major_event` | 直近の重要イベントからの営業日数 |
| `event_type_next` | 次のイベント種別（カテゴリカル: FOMC, BOJ, NFP, CPI, ...） |
| `is_event_day` | 当日が何らかの重要イベント日なら1 |
| `event_density_past_5d` | 過去5営業日以内のイベント数（イベント集中度。未来情報を含めない） |

### 管理方法

- NFP のように規則性があるものはルールベースで自動生成（第1金曜日）
- FOMC・日銀のように不規則なものは年次リストで手動管理
- `data/events.py` 内に日程リストを定義
- `event_density_past_5d` 等のイベント特徴量は decoder_data 生成時にも利用されるため、カレンダーリストを最新状態に保つことが予測品質に直結する。年次更新を怠ると decoder 側の特徴量が 0 で固定され、予測精度が劣化する

## モデル構成 (`model/`)

### TFT パラメータ

| パラメータ | 値 | 備考 |
|---|---|---|
| encoder_length | 60 | 入力ウィンドウ（営業日） |
| prediction_length | 5 | 予測ホライズン |
| hidden_size | 64 | 過学習防止のため小さめ |
| attention_head_size | 4 | TFT論文準拠 |
| dropout | 0.2 | |
| hidden_continuous_size | 32 | |
| quantiles | [0.1, 0.25, 0.5, 0.75, 0.9] | 分位点予測 |
| output_size | 5 | quantiles の要素数に合わせて明示指定 |

### カスタム損失関数 (`model/loss.py`)

`DirectionAwareQuantileLoss`: QuantileLoss にスムーズな方向ペナルティを追加。

- ベース: `QuantileLoss`（分位点予測）
- 追加: 中央値予測と実績の方向不一致にペナルティ。`tanh(pred / temperature)` によるスムージングを適用し、ゼロ付近の勾配不連続を回避
- `direction_weight` と `smoothing_temperature` を `config.py` で調整可能
- ゼロ付近のデッドゾーン幅（`dead_zone`）をパラメータ化し、微小リターンでのペナルティ振動を抑制

### 学習パイプライン (`model/trainer.py`)

| 項目 | 設定 |
|---|---|
| フレームワーク | PyTorch Lightning |
| Early Stopping | val_loss, patience=10 |
| LR Scheduler | ReduceLROnPlateau |
| チェックポイント | val_loss top-3 を保存 |
| GPU対応 | TF32 + cuDNN autotune（CUDA利用時） |
| データ分割 | 時系列順に train:val:threshold_tune:test = 6.5:1.5:0.5:1.5（注: threshold_tune は約125営業日だが、セット境界で prediction_length=5 の境界効果があるため実質独立評価期間は約120営業日） |

### 学習実行 (`scripts/train.py`)

```
uv run python scripts/train.py                # 通常学習
uv run python scripts/train.py --optuna       # Optunaハイパラチューニング
```

Optuna モードでは `hidden_size`, `dropout`, `learning_rate`, `direction_weight` 等を探索。
結果は `artifacts/optuna_study.db` に保存。

## 評価 (`scripts/evaluate.py`)

### メトリクス

| メトリクス | 説明 |
|---|---|
| QuantileLoss | 分位点予測の精度 |
| MAE / RMSE | 中央値予測の誤差 |
| 方向精度 | 各ホライズン(1d〜5d)の上昇/下降的中率 |
| シャープレシオ | 予測に基づく仮想トレードの収益性 |
| キャリブレーション | 分位点の信頼区間カバー率 |
| `actual_up_ratio` | 評価期間中の実際の上昇日比率 |
| `pred_up_ratio` | モデルが上昇と予測した比率 |
| `direction_ratio_gap` | `abs(actual_up_ratio - pred_up_ratio)` 各ホライズン別 |

### 方向比率キャリブレーション

モデルの予測 up:down 比率が実績の比率と乖離しないようにする仕組み:

1. **学習時:** `DirectionAwareQuantileLoss` の方向ペナルティで曖昧な予測を抑制
2. **評価時:** `direction_ratio_gap` を主要メトリクスとして追跡
3. **推論時:** 動的閾値チューニングで比率を補正

#### 動的閾値チューニング

固定閾値(0)ではなく、**閾値チューニング専用セット**で最適閾値を探索:

```
予測値 > threshold → 上昇予測
予測値 ≤ threshold → 下降予測
```

- データ分割を train:val:threshold_tune:test = 6.5:1.5:0.5:1.5 に変更
- バリデーションセット: モデル選択（Early Stopping、チェックポイント）専用
- 閾値チューニングセット: `threshold_tune` 期間の up:down 比に最も近くなる threshold を各ホライズンごとに算出
- テストセット: 閾値チューニングから完全に独立した最終評価用
- 算出した閾値を保存し、推論時・ダッシュボードで使用

### 出力

- top-3 チェックポイントのアンサンブルで評価
  - 点予測（q50）: 3モデルの中央値予測を平均
  - 信頼区間: q10 は3モデルの min、q90 は3モデルの max を採用（保守的な区間推定でカバレッジを維持）
- `artifacts/eval_report.json` + 可視化グラフ(`.png`)

## 日次予測 (`scripts/predict.py`)

処理フロー:
1. **encoder_data 構築:** 最新60営業日分の全特徴量（observed + unknown + known）を取得・計算
2. **decoder_data 構築:** 予測対象5営業日分の `time_varying_known` 特徴量を生成
   - カレンダー: 未来5日の曜日, 月, 月末フラグを `pd.bdate_range` で生成
   - イベント: `data/events.py` の日程リストから各特徴量を算出（イベントカレンダーは事前公表済みのため decoder でも計算可能）:
     - `days_to_next_major_event`: decoder の各未来ステップ t において、t 日を基準日としてカレンダーから次イベントまでの営業日数を算出
     - `event_type_next`: decoder の各未来ステップ t において、t 日以降で最初に到来するイベントの種別を算出
     - `event_density_past_5d`: decoder の各未来ステップ t において、`t-5d` 〜 `t-1d` の範囲のイベント数をカレンダーリストから算出
     - `days_from_last_major_event`, `is_event_day`: 同様に t 日基準で算出
3. encoder_data + decoder_data を結合し `TimeSeriesDataSet` に渡して `model.predict()` を実行
4. 結果を `artifacts/predictions.db`（SQLite）に蓄積
5. 出力: 各ホライズンの中央値予測 + 信頼区間(10%-90%)
6. 動的閾値を適用した方向シグナル付き

## ダッシュボード (`dashboard/app.py`)

Streamlit で以下を1画面に表示:

| パネル | 内容 |
|---|---|
| 予測チャート | USD/JPY の実績 + 5日先予測（信頼区間付きファンチャート） |
| 方向シグナル | 各ホライズンの上昇/下降予測とその確信度 |
| 方向比率モニター | 直近N日の実績 up 比率 vs 予測 up 比率の折れ線グラフ |
| 特徴量重要度 | TFT の Variable Selection Network が選んだ重要特徴量 |
| Attention | TFT の時間方向 Attention ヒートマップ |
| イベントカレンダー | 今後の FOMC・雇用統計等のスケジュール |
| 過去予測の精度 | 直近30日の予測 vs 実績の方向精度推移 |

```
uv run streamlit run dashboard/app.py
```

## 定期実行

Windows タスクスケジューラで `scripts/predict.py` を毎営業日の日本時間 7:00（NY市場クローズ後）に実行。
`scripts/schedule_task.bat` でセットアップ。予測結果は SQLite に蓄積され、ダッシュボードから参照。

## 依存パッケージ

```toml
[project]
requires-python = ">=3.11"
dependencies = [
    "pytorch-forecasting",
    "lightning",
    "torch",
    "yfinance",
    "fredapi",
    "python-dotenv",
    "ta",
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn",
    "optuna",
    "streamlit",
    "plotly",
]
```

パッケージ管理は `uv` を使用。
