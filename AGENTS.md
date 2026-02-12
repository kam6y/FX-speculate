# Codex Agent Settings for FX-Speculate Project

このファイルは、Codexに渡される共通のコーディング規約や設定を記述します。
ClaudeからCodexに委譲される際、毎回この規約が適用されます。

## プロジェクト概要
- FX取引（USD/JPY）の機械学習モデル構築プロジェクト
- Python 3.11 / パッケージ管理: uv（pyproject.toml + uv.lock）
- 主要ライブラリ: pandas, numpy, scikit-learn, CatBoost, Optuna, SHAP

## プロジェクト構造

| ファイル | 役割 |
|---|---|
| `米ドルFxモデル構築_v_5.ipynb` | モデルv5（ローカル実行対応版） |
| `米ドルFxモデル構築_v_6.ipynb` | モデルv6（最新の開発版） |
| `usd_jpy_1min_*.parquet` | USD/JPY 1分足データ（読み取り専用、変更不可） |
| `pyproject.toml` / `uv.lock` | 依存関係管理 |
| `AGENTS.md` | Codex向け規約（このファイル） |
| `CLAUDE.md` | CC向け運用ルール |

## コーディング規約

### 基本スタイル
- PEP 8準拠、行の長さ最大100文字、インデント4スペース
- 命名: 変数・関数は`snake_case`、クラスは`PascalCase`、定数は`UPPER_SNAKE_CASE`

### FX時系列データの鉄則
- **look-ahead bias禁止**: 未来のデータを特徴量・前処理に使わない
- **データリーク防止**: クロスバリデーション分割は時系列順を維持する（TimeSeriesSplit等）
- **forward looking禁止**: バックテストで未来情報を参照しない
- 時刻はUTC基準で統一する

### 特徴量エンジニアリング
- 特徴量生成は関数化し、学習・推論で同一関数を使う
- ローリング計算のwindowサイズは定数で定義する
- 新しい特徴量追加時は、既存の特徴量との相関を確認する

### モデル学習
- ハイパーパラメータは関数引数またはセル先頭の定数で明示する
- random seedは固定する（再現性確保）
- CatBoostのGPU使用は`USE_GPU`フラグで制御し、CUDA不可時はCPUにフォールバックする
- Optuna試行数・タイムアウトは定数で制御する

### パフォーマンス
- データはparquet形式で読み書きする
- ボトルネック処理はNumba JITまたはvectorized操作を使う
- 不要な`.copy()`を避け、メモリ効率を意識する

## レビュー基準

Codexがレビュー役を担う際の緊急度定義:

| 緊急度 | 定義 | 例 |
|---|---|---|
| **high** | データリーク・look-ahead bias・計算結果の誤り・セキュリティ問題 | 未来データで特徴量生成、テストデータの混入 |
| **medium** | パフォーマンス劣化・可読性低下・規約違反 | 不要なループ、命名規則違反 |
| **low** | 改善提案・スタイルの軽微な指摘 | コメント追加の提案、変数名の改善 |

- highが1つでもあれば修正必須（CLAUDE.mdの品質基準と連動）
- medium以下は報告のみ、修正は任意
