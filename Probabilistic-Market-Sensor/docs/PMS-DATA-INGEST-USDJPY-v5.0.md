# PMS データ取得・保存規約 v5.0（USDJPY）

## 1. 目的
USDJPYの1分足OHLCVとマクロイベントを、再現可能かつ最小構成で取り込み・保存するための基準を定義する。
本ドキュメントはWBSステップ2を実装し、PMS-REQ-USDJPY-v5.0 に整合する。

## 2. 参照元（ソースオブトゥルース）
- PMS-REQ-USDJPY-v5.0.md: 6.1（OHLCV）、6.2（マクロイベント）、13.2（infer入力制約）
- PMS-REQ-USDJPY-v5.0-WBS.md: ステップ2（データ取得・保存基盤）

## 3. 未決事項（要確認）
以下はPMS-REQ-USDJPY-v5.0に記載がないため、運用前に確定が必要。
- OHLCVデータ提供元（ベンダー/API、ライセンス、レート制限）
- マクロイベント提供元と、OpenAPIに記載するevent_typeの列挙リスト
- 更新方式（バッチ or インクリメンタル）と保持ポリシー（raw/curated）
- リポジトリ外の保存先ルール（ローカル or オブジェクトストレージ等）

## 4. 正規化ルール（固定）
- timezone: UTC固定
- timestamp形式: ISO8601 + Zサフィックス、秒=0（例：2026-01-26T12:34:00Z）
- 配列順: 古い→新しい
- symbol: USDJPY（スクリプトで固定チェック）
- timeframe_sec: 60（スクリプトで固定チェック）

## 5. 保存形式とレイアウト（ベースライン）
- 形式: JSON Lines（JSONL）
- メタデータ: ファイルごとにmanifest JSON（既定はデータファイルと同じディレクトリ。`--manifest-dir` で分離可）
- 推奨ディレクトリ（デフォルトでは作成しない）:
  - data/curated/ohlcv/
  - data/curated/macro_events/
  - data/manifests/（`--manifest-dir` を指定する場合）

### 5.1 ファイル命名
- OHLCV: ohlcv_usdjpy_1m_YYYYMMDD_YYYYMMDD.jsonl
- マクロイベント: macro_events_YYYYMMDD_YYYYMMDD.jsonl
- Manifest: <同一プレフィックス>.manifest.json

## 6. スキーマ（固定）

### 6.1 OHLCVレコード（JSONL）
フィールド（JSONの順序は問わないが、出力は正規化する）:
- timestamp_utc（string, ISO8601 UTC, 秒=0, Zサフィックス）
- open（float）
- high（float）
- low（float）
- close（float）
- volume（float）

### 6.2 マクロイベントレコード（JSONL）
フィールド:
- event_type（string）
- scheduled_time_utc（string, ISO8601 UTC, 秒=0, Zサフィックス）
- importance（string, low/medium/high）
- revision_policy（string, none/revision_possible/revision_expected）
- published_at_utc（string, optional）
- actual（number, optional）
- forecast（number, optional）
- previous（number, optional）
- unit（string, optional）

### 6.3 Manifest（JSON）
共通キー:
- dataset（"ohlcv" または "macro_events"）
- symbol（USDJPY）
- timeframe_sec（OHLCVのみ）
- timezone（UTC）
- records（件数）
- start_timestamp_utc / end_timestamp_utc（OHLCV）
- start_scheduled_time_utc / end_scheduled_time_utc（マクロ）
- source（自由記述）
- generated_at_utc
- file（データファイル名）
- schema（フィールド一覧と順序方針）

## 7. 取り込みスクリプト雛形
プロバイダ非依存の最小雛形を提供する。
- scripts/ingest_usdjpy.py
- サブコマンド:
  - ohlcv: OHLCVの正規化・保存
  - macro: マクロイベントの正規化・保存

### 7.1 使用例（実行はしない）
```bash
python scripts/ingest_usdjpy.py ohlcv --input raw_ohlcv.csv --format csv --output-dir data/curated/ohlcv --source VendorA
python scripts/ingest_usdjpy.py macro --input raw_macro.jsonl --format jsonl --output-dir data/curated/macro_events --source VendorB
```

## 8. 最低保存要件
- テスト期間要件に合わせ、1分足OHLCVは最低3か月分を保存できること。
- スクリプトの警告判定は「1分足バー数（min_coverage_days * 24 * 60）」を基準とする。

## 9. バリデーション範囲
- 本ベースラインはタイムスタンプ正規化と順序保証を行う。
- リーク防止チェックと event_type の許可リスト検証は、提供元とenum確定後に実装する。

