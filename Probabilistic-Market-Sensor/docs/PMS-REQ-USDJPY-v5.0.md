# 市場環境認識AIモジュール（Probabilistic Market Sensor）要件定義書 v5.0（10m/30m/60m 並列）

- 文書ID：PMS-REQ-USDJPY-v5.0
- 作成日：2026-01-26
- 対象：USD/JPY
- 用途：トレード戦略を決定するLLMエージェントへ「確率分布＋根拠＋類似事例＋リスク旗」を供給するセンサーモジュール
- 注記：本モジュールは売買執行・収益保証を行わない（PF検収はスコープ外）

## 1. 目的

USD/JPYの市場データおよびマクロイベント情報から、10分先・30分先・60分先の市場状態を確率的に推定し、LLMが意思決定できるよう以下を低遅延APIで返す。

- 方向性確率（4クラス）× 3ホライズン
- 価格レンジ（分位点 q10/q50/q90）× 3ホライズン
- 類似局面検索（過去事例とその後の展開：10/30/60の結果を含む）
- 予測根拠（寄与特徴・寄与時間：ホライズン別）
- OOD/データ品質のリスク旗（入力共通）

## 2. スコープ

### 2.1 スコープ内

- データ前処理・特徴量生成
- 学習パイプライン（再学習含む）
- 推論エンジン（GPU最適化含む）
- 類似局面検索（Embedding生成＋検索I/F）
- Explainability（説明）生成（ホライズン別）
- OOD検知（入力共通）
- APIサービング、監視、ログ設計
- デバッグ可視化（SLO分離）
- OpenAPI仕様書作成（必須）

### 2.2 スコープ外

- 注文執行（取引所/ブローカー接続）
- ポジション管理、資金管理、PF保証
- LLM側プロンプト設計（I/F整合は実施）

## 3. 用語

- OHLCV：Open/High/Low/Close/Volume
- Horizon：予測対象の将来時点（本書では 10分/30分/60分）
- ECE：Expected Calibration Error
- OOD：Out-of-Distribution
- Pure Inference Time：前後処理/I/Oを除くGPU推論時間
- E2E Latency：API受信〜応答まで（前後処理・検索・I/O含む）
- asof_timestamp_utc：推論基準時刻（バー確定時刻）

## 4. 前提条件・制約（固定）

### 4.1 対象ハードウェア／ソフトウェア（前提固定）

#### GPU（前提固定）

- GPU：NVIDIA GeForce RTX 5070 Ti（Blackwell世代）を最低ターゲットとする（本番および検収の前提）。
- VRAM：16GB GDDR7 構成を前提とする（推論常駐モデル＋バッファを確保）。
- 推論精度：FP8 / FP16 / BF16 のいずれかを採用可能とする。採用精度は model_version とともに文書（Runbookまたはモデルカード）へ明記する。
- Blackwellの精度モード（FP8等）を前提に設計可能だが、最終的な検収は納品時点の実装スタック（ドライバ/CUDA/TensorRT等）で成立したモードをもって行う。

#### OS

- OS：Ubuntu 22.04/24.04 または Windows 11（最終確定は納品形態で決定）
- 標準検収OS：Ubuntu（推奨）

#### コンテナ

- Dockerによる標準納品を必須とする。
- GPU実行に必要なランタイム（nvidia-container-toolkit等）を前提とした起動手順を成果物に含める。

#### 責任分界（固定）

- 甲（発注側）は検収に用いる機材として RTX 5070 Ti（16GB）を準備する。
- 乙（受注側）は当該GPU上で、本書のレイテンシ・品質要件を満たす推論サーバ／モデルを納品する。
- 甲都合で検収GPUが異なる場合は、性能検収条件の変更として別途合意する。

### 4.2 レイテンシ制約（検収条件：3ホライズン同時）

本モジュールは 1回のinferで 10/30/60分を同時返却する（13章参照）。
通常 infer（debug=false, horizons=[10,30,60]）について以下を満たすこと。

#### Pure Inference Time

- p50 ≤ 25ms、p99 ≤ 40ms

#### E2E Latency

- p50 ≤ 45ms、p99 ≤ 80ms

#### 計測条件（固定）

- GPU：RTX 5070 Ti 16GB（Blackwell）
- バッチサイズ=1
- lookback_bars=512（デフォルト）
- モデルGPU常駐（ロード済み・ウォームアップ済み）
- 計測回数：10,000リクエスト以上でp99算出
- 同一マシン上での並列度：同時4リクエストまで性能劣化許容（同時4でE2E p99 ≤ 120ms）
- 類似検索（ベクトルDB）は標準検収条件として同一マシン内（ローカル）

#### debugの扱い

- debugは通常SLO対象外（12.2参照）。

## 5. システム全体構成（論理）

- Ingestion：API入力としてOHLCV＋マクロイベントを受領
- Feature Builder：欠損補完・特徴量生成・正規化
- Inference Core：ホライズン別の Direction/Range、共通Embedding、ホライズン別Explainability、共通OOD生成
- Similarity Search：ベクトルDB照会（TopK返却）
- API：LLMへJSON応答
- Observability：メトリクス・ログ・トレース・モデル版管理

## 6. 入力データ仕様（検収可能定義）

### 6.1 OHLCV（必須）

- symbol：USDJPY
- timeframe_sec：60（1分足）
- タイムゾーン：UTC固定
- 配列順：古い→新しい
- 制約：ohlcv.length == lookback_bars を必須

#### レコードスキーマ（1分）

- timestamp_utc：分の開始時刻（ISO8601 UTC、秒=0）
- open/high/low/close：float
- volume：float（取得不可の場合0許容。ただしvolume依存特徴を無効化し警告フラグを立てる）

#### 入力整合性（422）

- timestamp単調増加（重複禁止）
- high >= max(open,close)、low <= min(open,close)、high >= low
- NaN/inf禁止

#### 欠損補完

- 最大連続3本まで補完可：前値closeでOHLC埋め、volume=0
- 4本以上連続欠損：risk_flags.data_integrity_warning=true かつ risk_flags.degraded=true（推論は実施）

### 6.2 マクロイベント（必須）

（v4.1相当を踏襲。リーク防止は入力検証で担保。）

#### レコードスキーマ

- event_type、scheduled_time_utc、importance、revision_policy（必須）
- published_at_utc（任意：実績を載せる場合は必須）
- actual/forecast/previous（任意）
- unit（任意）

#### リーク防止（422）

- published_at_utc > asof_timestamp_utc なら422
- actual/forecast/previousがある場合 published_at_utc 必須
- 将来イベントに actual/forecast/previous が含まれる場合422

### 6.3 時刻特徴（必須）

- day_sin/cos、week_sin/cos をUTC基準で生成

## 7. 特徴量要件

### 7.1 必須特徴量（最小セット）

- 価格系：log return（close）、high-lowレンジ（正規化）
- 出来高系：log(volume+1)、volume zscore（ロバスト推奨）
- ボラ系：realized volatility（窓長デフォルト30）
- 時刻埋め込み：daily/weekly sin-cos
- マクロ近傍：event_onehot、event_decay_past、event_time_to_next

### 7.2 ロバスト性要件（必須）

- 方式A：winsorize（例：p0.5〜p99.5）または
- 方式B：median/MAD等のロバスト統計正規化

採用方式はFeature辞書に固定記載する。

## 8. 予測ターゲットとラベル定義（ホライズン別に固定）

### 8.1 Horizons（固定）

- H = 10分（10 bars）
- H = 30分（30 bars）
- H = 60分（60 bars）

### 8.2 バリア（4クラス、ホライズン別）

ホライズン H（分）ごとに、以下でラベルを定義する。

- 基準価格：C_t（t時点close）
- 上バリア：C_t + 0.5 * ATR20
- 下バリア：C_t - 0.5 * ATR20
- 判定窓：t+1〜t+H の各1分バーの high/low でヒット判定
- 同一バー内で上下両方ヒット：Choppy

#### クラス定義（各Hで同一）

- Up：上バリアのみヒット
- Down：下バリアのみヒット
- Choppy：上下両方ヒット
- Neutral：いずれもヒットしない

### 8.3 ATR20定義（固定）

- True Range：max(high-low, abs(high-prev_close), abs(low-prev_close))
- ATR20：SMA（単純移動平均）20本で固定する

## 9. モデル要件（要件＝出力と品質、実装＝自由）

### 9.1 必須出力（ホライズン別）

各ホライズン H∈{10,30,60} について必ず返す。

- Direction：4クラス確率（up/down/neutral/choppy）
- Range Forecast（固定）：close(t+H) の q10/q50/q90（価格単位）
- Explainability：dominant_factors Top5、critical_time_offsets_min Top5（分単位）
- horizon_min：10/30/60 を必ず付与

### 9.2 必須出力（共通）

- Similarity Embedding：128次元（生成必須。返却は設定可だが標準では返す）
- OOD：ood_score と is_ood（入力共通）
- risk_flags：data_integrity_warning / degraded / degraded_reasons（入力共通）

### 9.3 実装自由度

- Backbone（例：PatchTST等）は推奨に留め、差し替え可。
- 合否は第14章の検収指標を満たすこと。

### 9.4 キャリブレーション（ホライズン別）

- 方式は自由（Temperature Scaling等）
- 評価はホライズン別に ECE（bins=15、top-label）を算出する。

## 10. OOD検知要件（入力共通）

### 10.1 必須

- ood_score（連続値、大きいほどOOD）
- is_ood（bool）

### 10.2 本番方式（いずれかに固定）

- A：再構成誤差（AE系）
- B：Mahalanobis距離（Embedding空間）
- C：アンサンブル分散

採用方式を固定し文書化する。

### 10.3 閾値（固定）

- 学習期間内ood_score分布の p99.7 を閾値として固定する。

### 10.4 検収（固定）

- 合成ショック：±1.0% の瞬断（1分内ジャンプ）を注入した系列で評価
- 合否：TPR ≥ 0.70、FPR ≤ 0.05

## 11. 類似相場検索要件（3ホライズン整合）

### 11.1 目的

LLMに「過去の類似局面と、その後10/30/60分でどうなったか」を提示する。

### 11.2 検索定義

- クエリ窓：lookback_barsと同長
- 候補集合：Train+Val（Test除外）
- 類似度：cosine similarity
- 返却：TopK（K=10）

### 11.3 返却項目（各事例に必須）

- scenario_start_utc / scenario_end_utc
- similarity_score
- then_direction_class_10m / 30m / 60m（Up/Down/Neutral/Choppy）
- then_return_10m / 30m / 60m（close(t+H)/close(t)-1）
- then_max_favorable_excursion_10m / 30m / 60m（%）
- then_max_adverse_excursion_10m / 30m / 60m（%）

### 11.4 オフライン評価（検収）

#### 教師信号（固定）

- 同一クラス（30mラベル）かつ RV同分位帯（±1分位帯）

#### 指標（固定）

- Recall@10 がランダムベースラインを上回ること
- 3つ以上の異なる期間で再現（期間は検収開始前に固定）

## 12. Explainability要件（ホライズン別）

### 12.1 必須出力（各ホライズン）

- method：文字列（attention / integrated_gradients 等）
- dominant_factors：特徴名Top5（Feature辞書内の名称）
- critical_time_offsets_min：分単位Top5
- confidence_note：固定テンプレ短文

### 12.2 デバッグ可視化（SLO分離）

inferとは分離し、以下のいずれかで提供（乙が方式を選定し固定）。

- A) GET /v1/market-sensor/debug/{inference_id}?horizon_min=30 等
- B) inferで debug_artifact_url を返し、ホライズン指定で取得

debug取得の目標：p99 ≤ 500ms（通常infer SLO対象外）

## 13. API仕様（OpenAPI必須）

### 13.1 エンドポイント

- POST /v1/market-sensor/infer
- GET /v1/market-sensor/health
- GET /v1/market-sensor/version
- （選定）GET /v1/market-sensor/debug/{inference_id} または debug_artifact_url

### 13.2 infer リクエスト（必須/制約）

#### 必須

- schema_version："1.0"
- inference_id（任意：未指定時サーバ発番）
- symbol："USDJPY"
- timeframe_sec：60
- asof_timestamp_utc
- lookback_bars：デフォルト512（最小128、最大2048）
- ohlcv：長さlookback_bars、古い→新しい
- macro_events：最大200件
- horizons_min：配列（デフォルト [10,30,60]、許容値は {10,30,60} のみ）

#### 任意

- debug：bool（デフォルトfalse）

#### 制約

- payload size 最大2MB（超過は413）
- OHLC整合、timestamp整合、リーク防止違反は422

### 13.3 infer レスポンス（必須）

- schema_version
- inference_id
- model_version
- asof_timestamp_utc
- latency_ms：pure_infer / e2e（当該リクエスト全体）
- embedding_128（返却する場合）
- per_horizon：ホライズン配列（10/30/60の各要素を含む）
- horizon_min
- market_condition：probabilities、entropy、regime（任意）
- range_forecast：target="close_t_plus_{H}m"、q10/q50/q90
- explainability
- historical_analogy：topk=10（11.3の then_* を含む）
- risk_flags（共通）
- is_ood、ood_score
- data_integrity_warning
- degraded
- degraded_reasons（配列）

### 13.4 エラー仕様（固定）

- 400：必須フィールド欠落、型不正
- 401/403：認証/認可失敗
- 413：payload過大
- 422：OHLC不整合、timestamp不整合、lookback不足、リーク疑い、horizons_min不正
- 429：レート制限
- 503：モデル未ロード/リソース逼迫
- 500：内部エラー（追跡ID付与）

## 14. 品質保証・検収基準（Acceptance Criteria：ホライズン別＋総合）

### 14.1 データ分割（固定）

- Train/Val/Test：時系列分割（未来情報遮断）
- Purge/Embargo：最大ホライズン60分に合わせ、境界リーク排除
- テスト期間：最低3か月（検収開始前に固定）

### 14.2 Direction（4クラス：ホライズン別）

以下をホライズン別（10/30/60）に評価する。

#### 合格基準（30分：主基準）

- Balanced Accuracy ≥ 0.40
- Macro-F1 ≥ 0.35
- Up Recall ≥ 0.30、Down Recall ≥ 0.30
- ベースライン（常にNeutral、直近符号）を上回ること

#### 合格基準（10分・60分：副基準）

- Balanced Accuracy ≥ 0.38
- Macro-F1 ≥ 0.33
- Up Recall ≥ 0.28、Down Recall ≥ 0.28
- ベースラインを上回ること

#### 総合合否（固定）

30分が主基準を満たし、かつ 10分・60分がそれぞれ副基準を満たすこと。

#### 提出物

- 混同行列、期間別レポート（各ホライズン）

### 14.3 Range Forecast（分位点：ホライズン別）

各ホライズンで以下を満たす。

- Pinball Loss（q=0.1/0.5/0.9平均）が単純ベースラインを上回ること
- カバレッジ（テスト期間全体）
  - P(y ≤ q10) が 0.10 ± 0.03
  - P(y ≤ q90) が 0.90 ± 0.03

### 14.4 キャリブレーション（ホライズン別）

- ECE（bins=15、top-label）
  - 30分：< 0.05
  - 10分・60分：< 0.06

### 14.5 レイテンシ

- 第4.2のSLOを満たすこと（3ホライズン同時infer）
- 計測スクリプトと生ログ提出

### 14.6 OOD

- 第10.4のTPR/FPRを満たすこと
- 閾値・方式・再現手順提出

### 14.7 類似検索

- Recall@10 がランダムベースラインを上回り、複数期間で再現すること

## 15. 運用・監視要件（MLOps）

### 15.1 ログ（必須）

- inference_id、asof、model_version、latency、ood_score、risk_flags
- 予測結果はホライズン別にログ化（確率要約、entropy、range要約）
- 価格系列保存は別途ポリシー化（保存する場合は暗号化・権限管理）

### 15.2 メトリクス（必須）

- latency（p50/p95/p99）
- OOD率、data_integrity_warning率
- 予測確率分布ドリフト（KL等）※ホライズン別
- Range coverageドリフト ※ホライズン別

### 15.3 モデル管理（必須）

- model_versionを応答に含める
- ロールバック手順（直前安定版）をRunbookに明記

### 15.4 再学習（推奨）

- 週次または月次（別途合意）
- 再学習→評価→承認→デプロイのゲート設計（ホライズン別レポート必須）

## 16. セキュリティ要件

- 認証：mTLS または Bearer Token（いずれか必須）
- 秘密情報：Vault/環境変数、ハードコード禁止
- SBOM提出（推奨）
- 認証失敗・レート制限発火の監査ログ

## 17. 成果物（納品物）一覧（検収対象）

- ソースコード一式（ビルド・起動手順）
- 推論サーバ（Dockerイメージ）
- 学習コード＋再学習手順
- 学習済みモデル（重み、最適化済みエンジンがあれば併記）
- OpenAPI仕様書（必須）＋サンプル
- 計測スクリプト（精度・Range・ECE・OOD・レイテンシ：ホライズン別）
- テストレポート（第14章全項目：ホライズン別＋総合合否）
- 運用Runbook（障害対応、モデル更新、ログ/メトリクス）

## 18. 実装フェーズ計画（検収ゲート付き）

- Phase 1（データ/特徴）：品質チェック、欠損注入テスト、Feature辞書確定
- Phase 2（学習）：ベースライン提示 → 指標達成（10/30/60）
- Phase 3（最適化/サービング）：TensorRT等、p99達成、API/スキーマ固定
- Phase 4（ストレス/OOD）：合成ショック、欠損連続、異常値、イベント欠落試験

## 参考：レスポンス例（構造のみ）

（OpenAPIではこれを厳密スキーマ化する。）

```json
{
  "schema_version": "1.0",
  "inference_id": "uuid-v4",
  "model_version": "pms-2026-01-26-001",
  "asof_timestamp_utc": "2026-01-26T12:34:00Z",
  "latency_ms": { "pure_infer": 22.1, "e2e": 61.4 },
  "per_horizon": [
    {
      "horizon_min": 10,
      "market_condition": {
        "probabilities": { "up": 0.22, "down": 0.18, "neutral": 0.15, "choppy": 0.45 },
        "entropy": 1.26
      },
      "range_forecast": { "target": "close_t_plus_10m", "q10": 153.10, "q50": 153.18, "q90": 153.30 },
      "explainability": {
        "method": "integrated_gradients",
        "dominant_factors": ["rv_30", "return_1", "event_time_to_next"],
        "critical_time_offsets_min": [-1, -3, -8],
        "confidence_note": "Explanation is heuristic; use as supporting evidence only."
      }
    },
    { "horizon_min": 30, "...": "..." },
    { "horizon_min": 60, "...": "..." }
  ],
  "historical_analogy": {
    "topk": [
      {
        "scenario_start_utc": "...",
        "similarity_score": 0.93,
        "then_direction_class_10m": "UP",
        "then_direction_class_30m": "CHOPPY",
        "then_direction_class_60m": "DOWN",
        "then_return_10m": 0.001,
        "then_return_30m": -0.0004,
        "then_return_60m": -0.002
      }
    ]
  },
  "risk_flags": {
    "is_ood": false,
    "ood_score": 1.12,
    "data_integrity_warning": false,
    "degraded": false,
    "degraded_reasons": []
  }
}
```
