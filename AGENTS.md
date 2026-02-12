# Codex Agent Settings for FX-Speculate Project

このファイルは、Codexに渡される共通のコーディング規約や設定を記述します。
ClaudeからCodexに委譲される際、毎回この規約が適用されます。

## プロジェクト概要
- FX取引（USD/JPY）の機械学習モデル構築プロジェクト
- Python 3.11を使用
- 主要ライブラリ: pandas, numpy, scikit-learn, CatBoost, Optuna, SHAP

## コーディング規約

### 基本方針
- データサイエンスプロジェクトとして、可読性と再現性を最優先
- Jupyter Notebookとスクリプトの両方に対応できる構造
- 実験管理しやすいコード構成
- パフォーマンスを意識した実装（Numba、vectorization等）

### Python スタイル
- 命名規則:
  - 変数・関数: `snake_case`
  - クラス: `PascalCase`
  - 定数: `UPPER_SNAKE_CASE`
  - プライベート変数: `_leading_underscore`
- インデント: 4スペース
- 行の長さ: 最大100文字
- 型ヒントを積極的に使用（Python 3.11+の機能を活用）

### データ処理
- pandas DataFrameの操作では、チェーンメソッドを活用
- コピーオンライトを意識（不要な`.copy()`を避ける）
- メモリ効率を考慮（dtypeの最適化、chunked processing等）
- 時系列データの扱いに注意（look-ahead biasを避ける）

### 機械学習コード
- 特徴量エンジニアリングは関数化し、再利用可能に
- ハイパーパラメータは定数として明示的に定義
- モデルの学習・評価は関数に分離
- クロスバリデーション時のデータリークに注意
- GPU使用時はCUDA availability checkを実装

### コメント・ドキュメント
- 複雑なアルゴリズムや数式には必ず説明を追加
- 関数にはdocstringを記述（Args, Returns, Raises）
- Notebookセルにはマークダウンで目的を明記

### エラーハンドリング
- データ読み込み失敗時の適切な処理
- 数値計算エラー（inf, nan）のチェック
- GPU利用不可時のfallback実装

### テスト
- データ処理関数は単体テスト可能な構造に
- 入力データのバリデーション
- 出力結果の形式チェック

## プロジェクト固有の規則

### データパス管理
- データファイルパスは定数として定義（`DATA_PATH`）
- 相対パスではなく、設定可能なパスを使用

### 実験管理
- 実験結果は再現可能な形で記録
- random seedの固定を忘れない
- モデルのバージョン管理を意識

### パフォーマンス
- 大規模データ処理ではparquet形式を推奨
- ボトルネックとなる処理はNumbaでJITコンパイル
- vectorized操作を優先

### GPU使用
- CatBoostのGPU使用は`USE_GPU`フラグで制御
- CUDA利用不可時のgraceful degradation

## 実装時の注意事項
- 金融時系列データの特性を理解した実装
- 時間的な前後関係を破壊しない（データリーク防止）
- 過学習を防ぐための適切な検証手法
- バックテストの実装時はforward lookingを避ける
