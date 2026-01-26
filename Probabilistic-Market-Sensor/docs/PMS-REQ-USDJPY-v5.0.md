# 市場環境認識AIモジュール（Probabilistic Market Sensor）要件定義書 v5.0（10m/30m/60m 並列）

- 文書ID：PMS-REQ-USDJPY-v5.0
- 作成日：2026-01-26
- 対象：USD/JPY
- 用途：トレード戦略を決定するLLMエージェントへ「確率分布＋根拠＋類似事例＋リスク旗」を供給するセンサーモジュール
- 注記：本モジュールは売買執行・収益保証を行わない（PF評価はスコープ外）

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

- GPU：NVIDIA GeForce RTX 5070 Ti（Blackwell世代）を最低ターゲットとする（本番および性能評価の前提）。
- VRAM：16GB GDDR7 構成を前提とする（推論常駐モデル＋バッファを確保）。
- 推論精度：FP8 / FP16 / BF16 のいずれかを採用可能とする。採用精度は model_version とともに文書（Runbookまたはモデルカード）へ明記する。
- Blackwellの精度モード（FP8等）を前提に設計可能だが、最終的な評価は運用開始時点の実装スタック（ドライバ/CUDA/TensorRT等）で成立したモードをもって行う。

#### OS

- OS：Ubuntu 22.04/24.04 または Windows 11（最終確定は運用形態で決定）
- 標準評価OS：Ubuntu（推奨）
- Windows 11 で運用する場合は WSL2 + Docker Desktop（Linuxコンテナ）+ NVIDIA GPU サポートを前提とし、Linuxコンテナで稼働させる

#### コンテナ

- Dockerによる標準提供（運用形態）を必須とする。
- GPU実行に必要なランタイム（nvidia-container-toolkit等）を前提とした起動手順を成果物に含める。

#### 責任分界（固定）

- 開発者は評価に用いる機材として RTX 5070 Ti（16GB）を準備する。
- 開発者は当該GPU上で、本書のレイテンシ・品質要件を満たす推論サーバ／モデルを構築・運用する。
- 評価GPUが異なる場合は、性能評価条件の変更として別途決定する。

### 4.2 レイテンシ制約（評価条件：3ホライズン同時）

本モジュールは 1回のinferで 10/30/60分を同時返却する（13章参照）。horizons_min は固定で、指定する場合は [10,30,60] のみ許容する。
通常 infer（debug=false, horizons_min=[10,30,60]）について以下を満たすこと。

#### Pure Inference Time

- p50 ≤ 25ms、p99 ≤ 40ms

#### E2E Latency

- p50 ≤ 45ms、p99 ≤ 80ms

#### 計測条件（固定）

- GPU：RTX 5070 Ti 16GB（Blackwell）
- バッチサイズ=1
- lookback_bars=512（デフォルト）
- lookback_barsが512以外のケースはSLO保証外（ベストエフォート）とする
- モデルGPU常駐（ロード済み・ウォームアップ済み）
- 計測回数：10,000リクエスト以上でp99算出
- 同時4リクエスト評価は4並列ワーカーで等間隔投入し、各ワーカー2,500件以上（合計≥10,000）。E2Eはキューイングを含めて算出
- 同一マシン上での並列度：同時4リクエストまで性能劣化許容（同時4でE2E p99 ≤ 120ms）
- 類似検索（ベクトルDB）は標準評価条件として同一マシン内（ローカル）

#### debugの扱い

- debugは通常SLO対象外（12.2参照）。

## 5. システム全体構成（論理）

- Ingestion：API入力としてOHLCV＋マクロイベントを受領
- Feature Builder：欠損補完・特徴量生成・正規化
- Inference Core：ホライズン別の Direction/Range、共通Embedding、ホライズン別Explainability、共通OOD生成
- Similarity Search：ベクトルDB照会（TopK返却）
- API：LLMへJSON応答
- Observability：メトリクス・ログ・トレース・モデル版管理

## 6. 入力データ仕様（評価可能定義）

### 6.1 OHLCV（必須）

- symbol：USDJPY
- timeframe_sec：60（1分足）
- タイムゾーン：UTC固定
- 配列順：古い→新しい
- 制約：ohlcv.length == lookback_bars を必須

#### レコードスキーマ（1分）

- timestamp_utc：分の開始時刻（ISO8601 UTC、秒=0）
- open/high/low/close：float
- volume：float（取得不可の場合0許容。ただし volume 依存特徴は 0 に固定し、volume_missing_flag=1 を付与して特徴量次元を固定する。risk_flags.data_integrity_warning=true、risk_flags.degraded=true、degraded_reasons に volume_missing を追加）

#### 入力整合性（422）

- timestamp単調増加（重複禁止）。並び順が崩れている場合は422
- high >= max(open,close)、low <= min(open,close)、high >= low
- NaN/inf禁止
- asof_timestamp_utc は「直近確定バーの終端時刻」とし、最後の ohlcv.timestamp_utc は asof_timestamp_utc - timeframe_sec と一致する
- ohlcv.length == lookback_bars を満たさない場合は422
- timestamp の「欠損」は 6.1 欠損補完の範囲で許容し、422対象外

#### 欠損補完

- 欠損定義：隣接timestampの差分が timeframe_sec を超える箇所を欠損とみなす
- 取引休止時間帯（市場クローズ）は欠損判定から除外する（具体的な時間帯はデータ取得・保存規約に記載）
- 最大連続3本まで補完可：前値closeでOHLC埋め、volume=0
  - 補完が発生した場合は risk_flags.data_integrity_warning=true、risk_flags.degraded=true
  - degraded_reasons に ohlcv_gap_filled を追加
- 4本以上連続欠損：同様に補完し推論は実施
  - risk_flags.data_integrity_warning=true、risk_flags.degraded=true
  - degraded_reasons に ohlcv_gap_too_long を追加
- 最大連続欠損は60本まで許容し、61本以上は422
- 補完後の系列を特徴量生成およびラベル生成に使用する
- 補完により系列長が lookback_bars を超える場合は、先頭側を切り詰めて lookback_bars 本に揃える（末尾は asof に整合）

### 6.2 マクロイベント（必須）

（v4.1相当を踏襲。リーク防止は入力検証で担保。）
macro_events は必須フィールドだが、イベントが存在しない場合は空配列を許容する。

#### レコードスキーマ

- event_type、scheduled_time_utc、importance、revision_policy（必須）
- published_at_utc（任意：実績(actual)を載せる場合は必須。実績の公表時刻）
- actual/forecast/previous（任意）
- unit（任意）
- scheduled_time_utc / published_at_utc は ISO8601 UTC（秒=0、Zサフィックス必須）

#### 値域（固定）

- event_type：事前に定義した経済指標ID（例：CPI_US, FOMC_RATE_DECISION 等）。OpenAPIに列挙（enum）として固定する。
- importance：{low, medium, high} のいずれか（lower-case固定）。
- revision_policy：{none, revision_possible, revision_expected} のいずれか（lower-case固定）。

#### リーク防止（422）

- published_at_utc が存在する場合、published_at_utc <= asof_timestamp_utc（未来時刻は422）
- actual がある場合 published_at_utc 必須 かつ published_at_utc <= asof_timestamp_utc
- scheduled_time_utc > asof_timestamp_utc のイベントに actual が含まれる場合422
- forecast/previous は scheduled_time_utc が未来でも許容（published_at_utc は不要。ただし付ける場合は <= asof_timestamp_utc）

#### 空配列・検証失敗の扱い（固定）

- macro_events が空配列の場合は許容するが、risk_flags.data_integrity_warning=true、risk_flags.degraded=true とし、degraded_reasons に macro_events_empty を追加する
- macro_events が存在し、必須項目欠落／enum逸脱／リーク防止違反がある場合は 422 を返し、推論は実行しない
- macro_events フィールド自体の欠落や型不正は 400（13.4参照）

### 6.3 時刻特徴（必須）

- day_sin/cos、week_sin/cos をUTC基準で生成

## 7. 特徴量要件

### 7.1 必須特徴量（最小セット）

- 価格系：log return（close）、high-lowレンジ（正規化）
- 出来高系：log(volume+1)、volume zscore（ロバスト推奨）
- ボラ系：realized volatility（窓長デフォルト30）
- 時刻埋め込み：daily/weekly sin-cos
- マクロ近傍：event_onehot、event_decay_past、event_time_to_next
- 欠損マスク：volume_missing_flag（0/1）

#### 数式定義（固定）

- t は asof_timestamp_utc - timeframe_sec のバー（直近確定バー）
- log return：r_t = ln(close_t / close_{t-1})
- high-lowレンジ（正規化）：(high_t - low_t) / close_t
- realized volatility（RV_30）：sqrt( Σ_{i=t-29..t} r_i^2 )
- volume zscore（ロバスト推奨の例）：z_t = (volume_t - median) / (MAD * 1.4826)
- median/MAD は推論時は lookback_bars 範囲（t-lookback_bars+1..t）で算出する
- volume_missing_flag=1 の場合は log(volume+1)=0、z_t=0 とする
- event_onehot：asof_timestamp_utc 以降 24h 以内で最も近い scheduled_time_utc を持つイベントの event_type を one-hot（該当なしは全0）
- event_onehot 同時刻が複数ある場合は importance（high > medium > low）→event_type 辞書順で一意に選択
- event_decay_past：published_at_utc が存在し、かつ published_at_utc <= asof_timestamp_utc の最新イベントとの差分分数 delta_t から exp(-delta_t/tau) を算出（tau=60分、該当なしは0）
- event_time_to_next：次回 scheduled_time_utc までの分数（0〜1440にクリップ、該当なしは1440）

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
- 判定窓内で上下両方ヒット（順序不問、同一バー含む）：Choppy
- t は asof_timestamp_utc - timeframe_sec のバー（直近確定バー）

#### クラス定義（各Hで同一）

- Up：上バリアのみヒット
- Down：下バリアのみヒット
- Choppy：上下両方ヒット
- Neutral：いずれもヒットしない
- ラベル文字列は {up, down, neutral, choppy} の lower-case で固定する

### 8.3 ATR20定義（固定）

- True Range：max(high-low, abs(high-prev_close), abs(low-prev_close))
- ATR20：SMA（単純移動平均）20本で固定する（t-19..t の20本、欠損補完後系列を使用）

## 9. モデル要件（要件＝出力と品質、実装＝自由）

### 9.1 必須出力（ホライズン別）

各ホライズン H∈{10,30,60} について必ず返す。

- Direction（market_condition.probabilities）：4クラス確率（up/down/neutral/choppy の lower-case 固定）
- probabilities は各クラス 0〜1 の範囲で、総和が 1.0（許容誤差 ±1e-6）になること。
- Range Forecast（固定）：close(t+H) の q10/q50/q90（価格単位）
- q10 <= q50 <= q90 を必須とする（単調性が崩れる場合はサーバ側で単調化して返却）
- 単調化を適用する場合、評価・ログ・返却は単調化後の値で統一する（14.3参照）
- Explainability：dominant_factors Top5、critical_time_offsets_min Top5（分単位）
- horizon_min：10/30/60 を必ず付与

### 9.2 必須出力（共通）

- Similarity Embedding：128次元（生成必須。返却は設定可だが標準では返す）。返却しない場合は embedding_128 を省略する（null/空配列は使用しない）。
- OOD：ood_score と is_ood（入力共通）
- risk_flags：data_integrity_warning / degraded / degraded_reasons（入力共通）

### 9.3 実装自由度

- Backbone（例：PatchTST等）は推奨に留め、差し替え可。
- 合否は第14章の評価指標を満たすこと。

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

### 10.4 評価（固定）

- 合成ショック：±1.0% の瞬断（1分内ジャンプ）を注入した系列で評価
- 合否：TPR ≥ 0.70、FPR ≤ 0.05

## 11. 類似相場検索要件（3ホライズン整合）

### 11.1 目的

LLMに「過去の類似局面と、その後10/30/60分でどうなったか」を提示する。

### 11.2 検索定義

- クエリ窓：lookback_barsと同長
- 候補集合：Train+Val（Test除外）。さらに推論時は asof_timestamp_utc 以前の履歴のみを候補に含める（将来データ混入を禁止）。
- 運用時は Train+Val に加えて、ラベル確定済みの推論履歴をインデックスへ追加可（追加頻度・保持期間はRunbookに明記）。評価時は Train+Val のみを使用。
- 類似度：cosine similarity
- 返却：TopK（K=10）
- 同一窓除外：scenario_end_utc == t（t は asof_timestamp_utc - timeframe_sec のバー）の候補は除外する

### 11.3 返却項目（各事例に必須）

- scenario_start_utc / scenario_end_utc
- similarity_score
- then_direction_class_10m / 30m / 60m（up/down/neutral/choppy）
- then_return_10m / 30m / 60m（close(t+H)/close(t)-1）
- then_max_favorable_excursion_10m / 30m / 60m（%）
- then_max_adverse_excursion_10m / 30m / 60m（%）

#### MFE/MAE 定義（固定）

- 基準価格：close_t（シナリオ窓の終端バーの close）
- then_max_favorable_excursion_Hm：max_i((high_i - close_t) / close_t) * 100
- then_max_adverse_excursion_Hm：min_i((low_i - close_t) / close_t) * 100
- i は t+1 〜 t+H の 1分バー（判定窓と同一）

### 11.4 オフライン評価（評価）

#### 教師信号（固定）

- H∈{10,30,60} それぞれについて、同一クラス（Hラベル）かつ RV同分位帯（±1分位帯）
  - RVは7.1の realized volatility（窓長30）を使用
  - RV分位は Train+Val 全体で 0-100 の百分位に分割し、同一分位±1を許容

#### 指標（固定）

- H∈{10,30,60} それぞれで Recall@10 がランダムベースラインを上回ること（30mを主基準）
- 3つ以上の異なる期間で再現（期間は評価開始前に固定）
- Recall@10 評価では、クエリ窓と時間が重なる候補は除外する（embargo=最大ホライズン60分）

## 12. Explainability要件（ホライズン別）

### 12.1 必須出力（各ホライズン）

- method：文字列（attention / integrated_gradients 等）
- dominant_factors：特徴名Top5（Feature辞書内の名称）
- critical_time_offsets_min：asof の直近確定バー(t)からの相対分（負値/0のみ、範囲は [-lookback_bars+1, 0]）
- confidence_note：固定テンプレ短文
  - entropy は natural log（ln）で算出した Shannon entropy（4クラス）とする。
  - テンプレ文言：`Explanation is heuristic; use as supporting evidence only.`

### 12.2 デバッグ可視化（SLO分離）

inferとは分離し、以下のいずれかで提供（開発者が方式を選定し固定）。

- A) GET /v1/market-sensor/debug/{inference_id}?horizon_min=30 等
- B) inferで debug_artifact_url を返し、ホライズン指定で取得

debug取得の目標：p99 ≤ 500ms（通常infer SLO対象外）
debug_artifact_url は debug=true の場合に infer レスポンスで返却し、TTL=24h を標準とする（認証は通常APIと同等）。

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
- asof_timestamp_utc（ISO8601 UTC、秒=0）
- lookback_bars：デフォルト512（最小128、最大2048）
- ohlcv：長さlookback_bars、古い→新しい
- macro_events：最大200件（空配列可）
- horizons_min：配列（固定 [10,30,60]。省略可だが、指定する場合は [10,30,60] のみ許容）

#### 任意

- debug：bool（デフォルトfalse）

#### 制約

- payload size 最大2MB（JSONのデコード後サイズ）。超過は413
  - 2MB超過が見込まれる場合は lookback_bars / macro_events を削減する
  - 圧縮（例：Content-Encoding: gzip）を許容する場合は OpenAPI に明記する
- OHLC整合、timestamp単調性違反、NaN/inf、リーク防止違反は422
- asof_timestamp_utc と ohlcv 最終バーの整合が取れない場合は422
- 欠損補完ポリシー内の timestamp gap は422対象外（連続欠損上限超過は422）
- lookback_bars 範囲外（<128 または >2048）や macro_events 件数超過（>200）は422

### 13.3 infer レスポンス（必須）

- schema_version
- inference_id
- model_version
- asof_timestamp_utc
- latency_ms：pure_infer / e2e（当該リクエスト全体）
- embedding_128（返却する場合）
- debug_artifact_url（debug=true の場合のみ。12.2参照）
- per_horizon：ホライズン配列（10/30/60の各要素を必ず含む）。horizon_min は一意で昇順（10,30,60）に並べる。各要素は horizon_min を必須で含み、以下を内包する
  - market_condition：probabilities、entropy、regime（任意。定義は OpenAPI の enum に固定し、未実装の場合は省略）
  - range_forecast：target="close_t_plus_{H}m"、q10/q50/q90
  - explainability
- historical_analogy：topk=10（11.3の then_* を含む）
- is_ood、ood_score
- risk_flags（data_integrity_warning / degraded / degraded_reasons）

#### degraded_reasons（列挙）

- volume_missing
- ohlcv_gap_filled
- ohlcv_gap_too_long
- macro_events_empty

### 13.4 エラー仕様（固定）

- 400：トップレベル必須フィールド欠落、型不正、JSONパース不能
- 401/403：認証/認可失敗
- 413：payload過大
- 422：OHLC不整合、timestamp単調性違反/末尾不一致、連続欠損上限超過、lookback不足、lookback範囲外、macro_events 件数超過、macro_events レコード必須項目欠落/enum逸脱、リーク疑い、horizons_min不正
- 429：レート制限
- 503：モデル未ロード/リソース逼迫
- 500：内部エラー（追跡ID付与）

## 14. 品質保証・評価基準（ホライズン別＋総合）

### 14.1 データ分割（固定）

- Train/Val/Test：時系列分割（未来情報遮断）
- Purge/Embargo：最大ホライズン60分に合わせ、境界リーク排除
- テスト期間：最低3か月（評価開始前に固定）

### 14.2 Direction（4クラス：ホライズン別）

以下をホライズン別（10/30/60）に評価する。

#### 合格基準（30分：主基準）

- Balanced Accuracy ≥ 0.40
- Macro-F1 ≥ 0.35
- Up Recall ≥ 0.30、Down Recall ≥ 0.30
- ベースラインを上回ること（以下のうち高い方を基準にする）
  - Baseline-A：常に neutral を予測
  - Baseline-B：直近1分リターン（close_t/close_{t-1}-1）の符号で予測（|return_1| < 0.02% は neutral）

#### 合格基準（10分・60分：副基準）

- Balanced Accuracy ≥ 0.38
- Macro-F1 ≥ 0.33
- Up Recall ≥ 0.28、Down Recall ≥ 0.28
- ベースラインを上回ること（14.2と同様の基準）

#### 総合合否（固定）

30分が主基準を満たし、かつ 10分・60分がそれぞれ副基準を満たすこと。

#### 記録物

- 混同行列、期間別レポート（各ホライズン）

### 14.3 Range Forecast（分位点：ホライズン別）

各ホライズンで以下を満たす。

- Pinball Loss（q=0.1/0.5/0.9平均）が単純ベースラインを上回ること
  - 単純ベースライン：訓練期間の H 分リターン分布の分位（q10/q50/q90）を固定し、close_t * (1 + q_p) を各分位として出力（p∈{0.1,0.5,0.9}）
- カバレッジ（テスト期間全体）
  - P(y ≤ q10) が 0.10 ± 0.03
  - P(y ≤ q90) が 0.90 ± 0.03
- 単調化（9.1）を適用する場合、Pinball Loss/カバレッジは単調化後の出力で算出する

### 14.4 キャリブレーション（ホライズン別）

- ECE（bins=15、top-label）
  - 30分：< 0.05
  - 10分・60分：< 0.06

### 14.5 レイテンシ

- 第4.2のSLOを満たすこと（3ホライズン同時infer）
- 計測スクリプトと生ログの保存

### 14.6 OOD

- 第10.4のTPR/FPRを満たすこと
- 閾値・方式・再現手順を文書化

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

- 週次または月次（別途決定）
- 再学習→評価→承認→デプロイのゲート設計（ホライズン別レポート必須）

## 16. セキュリティ要件

- 認証：mTLS または Bearer Token（いずれか必須）
- 秘密情報：Vault/環境変数、ハードコード禁止
- SBOM作成（推奨）
- 認証失敗・レート制限発火の監査ログ

## 17. 成果物一覧（評価対象）

- ソースコード一式（ビルド・起動手順）
- 推論サーバ（Dockerイメージ）
- 学習コード＋再学習手順
- 学習済みモデル（重み、最適化済みエンジンがあれば併記）
- OpenAPI仕様書（必須）＋サンプル
- 計測スクリプト（精度・Range・ECE・OOD・レイテンシ：ホライズン別）
- テストレポート（第14章全項目：ホライズン別＋総合合否）
- 運用Runbook（障害対応、モデル更新、ログ/メトリクス）

## 18. 実装フェーズ計画（評価ゲート付き）

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
  "debug_artifact_url": "https://example.com/debug/...",
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
        "then_direction_class_10m": "up",
        "then_direction_class_30m": "choppy",
        "then_direction_class_60m": "down",
        "then_return_10m": 0.001,
        "then_return_30m": -0.0004,
        "then_return_60m": -0.002,
        "then_max_favorable_excursion_10m": 0.12,
        "then_max_favorable_excursion_30m": 0.25,
        "then_max_favorable_excursion_60m": 0.40,
        "then_max_adverse_excursion_10m": -0.08,
        "then_max_adverse_excursion_30m": -0.15,
        "then_max_adverse_excursion_60m": -0.22
      }
    ]
  },
  "is_ood": false,
  "ood_score": 1.12,
  "risk_flags": {
    "data_integrity_warning": false,
    "degraded": false,
    "degraded_reasons": []
  }
}
```
