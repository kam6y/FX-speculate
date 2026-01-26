# FX-speculate

本リポジトリは FX-speculate / Probabilistic Market Sensor（USDJPY）の設計資料を保管します。
現在の実装状況：ドキュメント + データ取得スクリプト雛形（データ/モデル等の実体は未作成）。

## ソースオブトゥルース
- `Probabilistic-Market-Sensor/docs/PMS-REQ-USDJPY-v5.0.md`
- `Probabilistic-Market-Sensor/docs/PMS-REQ-USDJPY-v5.0-WBS.md`

## 追加ドキュメント
- `Probabilistic-Market-Sensor/docs/PMS-DATA-INGEST-USDJPY-v5.0.md` データ取得・保存規約

## 開発ベースライン（要件 v5.0 4.1）
- GPU: NVIDIA GeForce RTX 5070 Ti（16GB GDDR7）を評価の最低ターゲットとする。
- OS: Ubuntu 22.04/24.04 を推奨。Windows 11 の場合は WSL2 + Docker Desktop（Linux コンテナ）+ NVIDIA GPU サポートで運用する。
- コンテナ: Docker 必須。GPU 実行は NVIDIA コンテナランタイム／ツールキット前提。
- 精度: FP8 / FP16 / BF16（Runbook またはモデルカードに明記）。
- 責任: 開発者が上記 GPU を準備し、当該ハードウェアで評価を実施する。

## 参照手順（要件 v5.0 4.1）
1) 実装作業前に、ハードウェア／OS／コンテナ構成が上記ベースラインに合致することを確認する。
2) 実行・テスト手順を追加する場合は、WSL2 + Docker Desktop（Linux コンテナ）+ NVIDIA GPU の手順を明記する。
3) ドライバ／CUDA／TensorRT の実際のバージョンは、スタック確定後に Runbook へ記録する。

## 現在の構成（作成済み）
- `Probabilistic-Market-Sensor/docs/` 要件定義、WBS、データ取得・保存規約。
- `Probabilistic-Market-Sensor/scripts/` データ取得スクリプト雛形。
- `Probabilistic-Market-Sensor/src/` 入力スキーマ/バリデーションなどの実装コード。
- `Probabilistic-Market-Sensor/tests/` 入力バリデーションのテストケース。

## テスト（暫定）
前提: WSL2 + Docker Desktop（Linux コンテナ）環境で Python を実行する。
実行例（Linux コンテナ内）:
```bash
python -m unittest discover -s Probabilistic-Market-Sensor/tests -p "test_*.py"
```

## リポジトリ構成案（今後追加予定）
- `Probabilistic-Market-Sensor/runbook/` 運用手順と環境ノート。
- `Probabilistic-Market-Sensor/specs/` OpenAPI とスキーマ関連成果物。
- `Probabilistic-Market-Sensor/models/` 学習済みモデルと最適化済み推論成果物。
- `Probabilistic-Market-Sensor/logs/` 評価およびレイテンシ計測ログ。



