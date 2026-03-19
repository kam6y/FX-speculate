# 3モデル比較設計書

## 概要

USD/JPY 1分足FX方向予測システムにおいて、3つの独立したMLモデルを学習・バックテストし、性能を比較する。各モデルは独立したノートブックで管理し、共通コードはモジュール化して共有する。`/ralph-loop`による独立改善を前提とした設計。

## モデル構成

| # | モデル | アルゴリズム | パイプライン |
|---|--------|-------------|-------------|
| 1 | CatBoost | 勾配ブースティング（CatBoost） | 共通パイプライン |
| 2 | LightGBM | 勾配ブースティング（LightGBM） | 共通パイプライン |
| 3 | Transformer | Transformer Encoder（PyTorch） | 独自パイプライン |

## プロジェクト構成

```
FX-speculate/
├── data/
│   └── usd_jpy_1min_*.parquet
├── scripts/
│   ├── __init__.py             # パッケージ化
│   ├── update_data.py          # 既存: データ取得
│   ├── data_loader.py          # NEW: 共通データ読み込み
│   ├── features.py             # NEW: 共通特徴量生成
│   └── evaluation.py           # NEW: 共通バックテスト・評価
├── notebooks/
│   ├── usd_jpy_catboost.ipynb  # モデル1: CatBoost
│   ├── usd_jpy_lightgbm.ipynb  # モデル2: LightGBM
│   └── usd_jpy_transformer.ipynb # モデル3: Transformer
├── artifacts/
│   ├── catboost/               # .cbm, selected_features.pkl, config.pkl
│   ├── lightgbm/               # .txt, selected_features.pkl, config.pkl
│   └── transformer/            # .pt, config.pkl
```

### インポート戦略

各ノートブックの先頭で以下を実行し、`scripts/`をパッケージとしてインポート可能にする:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))

from scripts.data_loader import load_data
from scripts.features import generate_features, create_target, purged_time_series_split
from scripts.evaluation import run_backtest, compute_metrics, plot_equity_curve
```

## 共通モジュール

### `scripts/data_loader.py`

```python
def load_data(data_dir: str = "data") -> pd.DataFrame:
    """最新のparquetファイルを自動検出して読み込み。
    検出方法: glob 'usd_jpy_1min_*.parquet' → ファイル名末尾の日付をパースし最新を選択。
    Returns: timestamp, ask_open/high/low/close, bid_open/high/low/close のDataFrame (UTC)
    Raises: FileNotFoundError if no parquet files found.
    """
```

### `scripts/features.py`

- **`ta`ライブラリ特徴量**: `ta.add_all_ta_features()`によるテクニカル指標一括生成（※TA-Libではなく純Python `ta`パッケージ）
- **カスタム特徴量**: MA乖離率(4種)、EMA乖離率(2種)、ATR比率、ボリンジャー幅
- **リターン特徴量**: return_1, 5, 15, 60
- **時間特徴量**: hour/minute/dayのsin/cos、セッションフラグ（東京/ロンドン/NY）
- **ターゲット生成**: 指定pips閾値でBuy/Hold/Sellラベル作成
- **Purged Time Series Split**: 学習・検証・テスト分割（gap=15分）

```python
def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """全特徴量を生成して返す。欠損カラムがある場合はValueError。"""

def create_target(df: pd.DataFrame, threshold_pips: float, horizon: int = 15) -> pd.Series:
    """Buy(1)/Hold(0)/Sell(2)のターゲットラベルを生成。"""

def purged_time_series_split(df: pd.DataFrame, train_ratio=0.6, val_ratio=0.2, gap_minutes=15):
    """Purged分割。戻り値: (train_df, val_df, test_df)"""
```

CatBoost/LightGBMはこのモジュールをフル活用。Transformerはデータ読み込みとテクニカル指標生成部分のみ使い、ウィンドウ化は独自に行う。

### `scripts/evaluation.py`

```python
def run_backtest(predictions: np.ndarray, timestamps: pd.DatetimeIndex,
                 price_df: pd.DataFrame, feature_df: pd.DataFrame,
                 config: dict) -> BacktestResult:
    """Bid/Askバックテスト。独立トレード方式（ポジション追跡なし、同時建玉制限なし）。
    config keys: spread_pips, slippage_pips, api_fee_rate, bad_hours, atr_threshold, position_size, horizon
    Raises: ValueError if no trades generated.
    """

def compute_metrics(result: BacktestResult) -> dict:
    """Sharpe Ratio, Calmar Ratio, Profit Factor, Win Rate, Max Drawdown, Total P&L, Total Trades"""

def plot_equity_curve(result: BacktestResult) -> None:
    """エクイティカーブを描画。"""
```

- コストモデル: スプレッド0.2pips（Bid/Ask使用時は不要） + スリッページ0.1pips + API取引手数料0.002%（片道）
- Bad Hoursフィルター（UTC 20-23時スキップ）
- ATRフィルター
- バックテストは各バーを独立トレードとして評価（ポジション追跡なし、最大同時建玉制限なし）。3モデルとも同一の簡易ベースで比較。

## モデル1: CatBoost

### ノートブック: `notebooks/usd_jpy_catboost.ipynb`

既存v6をリファクタリングし、共通モジュールを使う形に書き換え。ロジック自体は変更なし。

- **特徴量**: 共通モジュールから生成、Purged CVでTop40選択（時間特徴量は最大4個制限）
- **学習**: CatBoost MultiClassOneVsAll、クラス重み自動バランス
- **最適化**: Optuna 40トライアル、Trading Sharpe目的関数
  - 探索対象: learning_rate, depth, l2_leaf_reg, random_strength, bagging_temperature, border_count, threshold_pips, prob_threshold
- **評価**: 共通evaluation.pyでバックテスト
- **アーティファクト**: `artifacts/catboost/` — `model.cbm`, `selected_features.pkl`, `config.pkl`

## モデル2: LightGBM

### ノートブック: `notebooks/usd_jpy_lightgbm.ipynb`

CatBoostと同一パイプラインで、モデル部分のみLightGBMに差し替え。

- **特徴量**: CatBoostと完全同一（共通モジュール、Top40選択、時間特徴量最大4個制限）
- **学習**: LightGBM `multiclass`、クラス重み自動バランス
- **最適化**: Optuna 40トライアル、Trading Sharpe目的関数
  - 探索対象: learning_rate, num_leaves, max_depth, min_child_samples, reg_alpha, reg_lambda, subsample, colsample_bytree, threshold_pips, prob_threshold
- **評価**: 共通evaluation.pyでバックテスト
- **アーティファクト**: `artifacts/lightgbm/` — `model.txt`, `selected_features.pkl`, `config.pkl`

**CatBoostとの違い**: ハイパーパラメータの探索空間のみ異なる。データ・特徴量・評価はすべて同一条件。

## モデル3: Transformer

### ノートブック: `notebooks/usd_jpy_transformer.ipynb`

独自パイプラインで、時系列ウィンドウ入力のTransformerエンコーダモデル。

### 入力データ

- **生の価格系列**: Ask/Bid OHLC（8カラム）
- **テクニカル指標**: 共通features.pyから生成（`ta`ライブラリ + カスタム指標）
- **ウィンドウ化**: 各時点で過去N本分の特徴量を`(batch, seq_len, features)`のテンソルに整形
- **正規化**: ウィンドウ単位でStandardScaler（未来の情報リークを防止）

### モデルアーキテクチャ

- **Positional Encoding**: 正弦波ベース（時系列の位置情報を付与）
- **Transformer Encoder**: Multi-Head Self-Attention × L層
- **分類ヘッド**: 最終時点の出力 → Linear → 3クラス（Buy/Hold/Sell）

### データ分割

- 共通の`purged_time_series_split()`で60/20/20に分割した**後**、各セットをウィンドウ化する
- **Purge gap**: `predict_horizon(15分) + window_size` に拡大し、ウィンドウの先頭が隣接セットに漏れることを防止
- Optuna最適化は学習コストを考慮し、**単一のtrain/val split**で実施（Purged CVは使わない）

### 学習詳細

- **損失関数**: CrossEntropyLoss（バランスドクラス重み付き、CatBoost/LightGBMと同じ方式）
- **オプティマイザ**: AdamW
- **学習率スケジューラ**: CosineAnnealingLR
- **早期停止**: val lossを監視、patience=10
- **勾配クリッピング**: max_norm=1.0（Transformerの学習安定性のため）

### 最適化（Optuna）

- **ウィンドウサイズ**: 30〜180本
- **モデル構造**: d_model, nhead, num_layers, dim_feedforward, dropout
- **学習**: learning_rate, batch_size, epochs（早期停止あり）
- **閾値**: threshold_pips, prob_threshold
- **目的関数**: Trading Sharpe Ratio（CatBoost/LightGBMと同じ基準）

### フレームワーク

- PyTorch（シンプルさ優先）
- GPU利用可能時は自動でCUDA使用

### アーティファクト

- `artifacts/transformer/` — `model.pt`, `config.pkl`

## 3モデル比較基準

### 共通評価指標（evaluation.pyで計算）

| 指標 | 説明 |
|------|------|
| Sharpe Ratio | リスク調整後リターン（主要比較基準） |
| Total P&L | 総損益（円） |
| Win Rate | 勝率 |
| Profit Factor | 総利益 / 総損失 |
| Max Drawdown | 最大ドローダウン |
| Calmar Ratio | リターン / 最大ドローダウン |
| Total Trades | 総取引数 |

### 比較条件の統一

- **テスト期間**: 同一（Purged Splitの最後20%）
- **コストモデル**: スプレッド0.2pips（Bid/Ask使用時は不要） + スリッページ0.1pips + API取引手数料0.002%（片道）
- **フィルター**: Bad Hours（UTC 20-23時）+ ATRフィルター
- **ポジションサイズ**: 10,000 units
- **バックテスト方式**: 独立トレード（ポジション追跡なし）

### 独立改善のルール

各ノートブックで変更**可能**なもの:
- 特徴量の追加・選択数の変更（例: Top40→Top60）
- ハイパーパラメータの探索空間・トライアル数
- モデルアーキテクチャの調整
- 前処理・正規化の変更

**変更不可**（公平な比較を担保）:
- テスト期間（Purged Splitの最後20%）
- コストモデル（スプレッド0.2pips（Bid/Ask使用時は不要） + スリッページ0.1pips + API手数料0.002%）
- フィルター（Bad Hours + ATR）
- 評価指標の計算方法（共通evaluation.pyを使用）
- ターゲット定義方法（threshold_pipsの値は変更可、定義式は不変）

## 技術スタック

- Python 3.11、パッケージ管理: uv
- CatBoost, LightGBM: 既存依存関係
- PyTorch (`torch>=2.0,<3.0`): 新規追加。`pyproject.toml`に追加し、CUDA対応のためuv設定で`extra-index-url`を指定
- Optuna: 全モデル共通のハイパーパラメータ最適化
- `ta` (Technical Analysis library): テクニカル指標（※TA-Libではない）
- scikit-learn: Purged CV、評価指標、クラス重み計算
- SHAP: 特徴量重要度（CatBoost/LightGBM）
- Pandas, NumPy, PyArrow, Matplotlib, Seaborn

**注記**: XGBoostは`pyproject.toml`に存在するが、本比較では使用しない。

## リアルタイム推論

本設計は学習・バックテスト・比較に焦点を当てる。リアルタイム推論（GMO Coin FX APIとの接続）は比較で最良モデルが決定した後に、そのモデルのノートブックに追加する。
