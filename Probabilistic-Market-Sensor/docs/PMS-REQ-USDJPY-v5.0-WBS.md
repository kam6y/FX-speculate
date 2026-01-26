# PMS-REQ-USDJPY-v5.0 WBS (20 Steps)

本WBSは「個人開発・並行作業なし」を前提に、上から順に実施する。

## 1. 開発基盤とリポジトリ構成の確定
- 目的: 作業の土台を固める
- 作業: GPU/OS/コンテナ方針の明記（RTX 5070 Ti/Ubuntu推奨/WindowsはWSL2+Docker）、ディレクトリ構成の骨子を決める
- 成果物: 基本構成メモ、ディレクトリ案
- 完了条件: 参照手順がREADMEに追記される（要件定義書v5.0 4.1準拠）

## 2. データ取得・保存基盤の作成
- 目的: 学習/評価に必要なOHLCVとマクロイベントを保存できる状態にする
- 作業: 1分足（timeframe_sec=60）OHLCVとマクロイベントの取り込み雛形、UTC統一（ISO8601/Z/秒=0）、配列順は古い→新しいで固定、保存形式/パス決定
- 成果物: データ取得スクリプト、保存規約
- 完了条件: 最低3か月分（テスト期間要件）を保存できる

## 3. 入力スキーマとバリデーション実装
- 目的: 仕様通りの入力制約をAPIで保証する
- 作業: 400/413/422条件（トップレベル必須フィールド欠落・型不正・JSONパース不能は400、macro_events欠落/型不正は400、レコード必須項目欠落/enum逸脱やリーク防止違反は422）、payload 2MB制限（JSONデコード後）、schema_version固定、symbol/USDJPY・timeframe_sec=60固定、lookback_bars範囲(128-2048)、macro_events<=200、horizons_minは省略可（指定時は[10,30,60]のみ許容）、timestamp整合（単調増加、最後の ohlcv.timestamp_utc == asof_timestamp_utc - timeframe_sec）、OHLC整合（high>=max(open,close)、low<=min(open,close)、high>=low）、NaN/inf禁止、ohlcv.length==lookback_bars、時刻フォーマット検証（ISO8601 UTC/Z/秒=0：asof/ohlcv/macroのscheduled/published）、リーク防止（published_at_utc/actual/scheduled_time_utc 条件）、macro_events空配列時は risk_flags.data_integrity_warning=true / degraded=true / degraded_reasons に macro_events_empty を追加、degraded_reasons列挙（volume_missing/ohlcv_gap_filled/ohlcv_gap_too_long/macro_events_empty）のOpenAPI反映、macro_eventsレコードの必須項目とenum検証（event_type/scheduled_time_utc/importance/revision_policy）
- 成果物: バリデーション関数、テストケース
- 完了条件: 仕様通りにエラーが返る／空配列時のrisk_flagsが付与される（要件定義書v5.0 6.2/13.2/13.4準拠）

## 4. 欠損補完ロジック実装
- 目的: 欠損条件の埋めとrisk_flagsの付与
- 作業: 欠損定義（timestamp gap>timeframe_sec）、最大連続60本、補完は前値close埋め/volume=0、risk_flags.data_integrity_warning=true／risk_flags.degraded=true の付与、degraded_reasons管理（ohlcv_gap_filled/ohlcv_gap_too_long/volume_missing）。volume欠損（取得不可で0の場合）はギャップ有無に関わらず volume_missing を追加し、data_integrity_warning/degraded を立てる。欠損補完はバリデーションでtimestamp単調性が確認できた後に適用し、補完後の系列で特徴量/ラベル生成を行う。補完で系列長が lookback_bars を超えた場合は先頭側を切り詰め、末尾は asof に整合させる
- 成果物: 欠損補完モジュール
- 完了条件: 61本以上連続は422、1-3本/4-60本で理由分岐、補完後はlookback_bars本に整形

## 5. 特徴量辞書と生成モジュール作成
- 目的: 必須特徴量を固定し再現性を担保
- 作業: 価格/出来高/ボラ/時刻/マクロ特徴を定義（log return、HLレンジ、RV_30、volume log/z、day/week sin-cos、event_onehot/decay/time_to_next、volume_missing_flag）、day/week sin-cos はUTC基準、event_onehotは asof_timestamp_utc 以降24h以内の最も近い scheduled_time_utc のイベントのみ（該当なしは全0）、同時刻タイブレーク規則（importance>辞書順）、event_decay_pastは published_at_utc <= asof_timestamp_utc の最新イベントからの delta_t で算出（tau=60分、該当なしは0）、event_time_to_nextのクリップ（0-1440分・該当なしは1440）を明記、volume_missing_flag=1時はlog/z=0固定、zscoreのmedian/MAD*1.4826 を明記、median/MADは推論時に lookback_bars 範囲（t-lookback_bars+1..t）で算出、ロバスト正規化方式を固定
- 成果物: Feature辞書、生成コード
- 完了条件: 任意の入力で特徴量が生成できる（要件定義書v5.0 7章準拠）

## 6. ラベル生成実装（10/30/60）
- 目的: 4クラスラベルを仕様通りに作る
- 作業: H=10/30/60分（各10/30/60 bars）、True Range= max(high-low, abs(high-prev_close), abs(low-prev_close))、ATR20(SMA)と±0.5*ATR20バリア、判定窓(t+1..t+H)のhigh/lowでup/down/choppy/neutral（同一バー両ヒット含む）、ラベル文字列はlower-case固定
- 成果物: ラベリングモジュール
- 完了条件: サンプル系列で手計算一致（要件定義書v5.0 8章準拠）

## 7. データ分割（時系列/Purge/Embargo）
- 目的: リーク防止の学習・評価を実現
- 作業: Train/Val/Test時系列分割、Purge/Embargo=最大ホライズン60分、テスト期間>=3か月
- 成果物: 分割スクリプト
- 完了条件: 分割再現性が担保される（要件定義書v5.0 14.1準拠）

## 8. ベースライン学習・評価
- 目的: 合格基準比較用のベースライン確立
- 作業: Baseline-A（常にneutral）、Baseline-B（1分リターン符号、|return_1|<0.02%はneutral）、レンジ用単純分位ベースライン
- 成果物: ベースライン評価レポート
- 完了条件: ベースライン指標が算出される（要件定義書v5.0 14.2/14.3準拠）

## 9. 本命モデル設計・学習
- 目的: 10/30/60同時出力モデルを確立
- 作業: 方向性4クラス確率/分位点レンジ/Explainability/Embedding128/OODの出力を定義し学習
- 成果物: 学習済みモデル、学習ログ
- 完了条件: per_horizonと共通出力が仕様通り得られる（要件定義書v5.0 9章準拠）

## 10. キャリブレーション実装
- 目的: ECE基準を満たす確率出力に調整
- 作業: Temperature Scaling等、ECE(bins=15, top-label)のホライズン別算出
- 成果物: キャリブレーションコード
- 完了条件: 30分<0.05、10/60分<0.06を判定できる

## 11. 分位点単調化処理の実装
- 目的: q10<=q50<=q90の保証
- 作業: 単調化ロジック、評価/ログ/返却の統一
- 成果物: 単調化ユーティリティ
- 完了条件: すべてのホライズンで単調性維持（要件定義書v5.0 9.1/14.3準拠）

## 12. OOD検知方式の選定・実装
- 目的: OOD検知の固定方式を確立
- 作業: 方式A/B/Cの選択、学習期間内分布のp99.7閾値固定、合成ショック(±1.0%)評価
- 成果物: OOD推論モジュール
- 完了条件: TPR>=0.70、FPR<=0.05の合否判定が可能

## 13. Embedding生成の実装
- 目的: 類似検索とOODに使う128次元埋め込み
- 作業: 推論時Embedding128生成（標準は返却。返却しない場合は省略し、null/空配列は使用しない）
- 成果物: Embedding生成コード
- 完了条件: 埋め込みが安定生成される（要件定義書v5.0 9.2準拠）

## 14. 類似局面検索実装
- 目的: TopK類似事例の提供
- 作業: cos類似度、TopK=10、クエリ窓はlookback_barsと同長、候補はTrain+Valのみ（Test除外、asof以前のみ）。運用時は Train+Val に加えてラベル確定済みの推論履歴を追加可（追加頻度・保持期間はRunbookに明記）、scenario_end==t除外、評価時はクエリ窓と時間が重なる候補を除外（embargo=60分）、教師信号は同一クラス+RV分位帯（±1分位）で定義、返却項目（scenario_start_utc/scenario_end_utc/similarity_score、then_direction_class_10/30/60m、then_return_10/30/60m、then_max_favorable_excursion_10/30/60m、then_max_adverse_excursion_10/30/60m）とMFE/MAE定義を固定、評価期間は開始前に固定し3期間以上で再現確認
- 成果物: 検索I/F、評価指標
- 完了条件: Recall@10がランダムベースライン超過を確認できる（30分を主基準、複数期間で再現）

## 15. Explainability実装
- 目的: ホライズン別根拠提示
- 作業: method/重要特徴Top5/時間オフセットTop5（分・負値/0のみ、範囲は[-lookback_bars+1, 0]）、entropyはlnのShannon entropyで算出、固定テンプレ文言（"Explanation is heuristic; use as supporting evidence only."）を付与
- 成果物: Explainability出力
- 完了条件: テンプレ文言が含まれる（要件定義書v5.0 12章準拠）

## 16. 推論コア統合・最適化
- 目的: 3ホライズン同時推論を最適化
- 作業: GPU常駐、バッチ1最適化、FP16/FP8/BF16検討、10,000件以上でp99算出、同時4リクエスト評価（4並列ワーカーで等間隔投入・各2,500件以上、E2Eはキューイング込みで算出）、計測条件（lookback_bars=512固定、lookback_barsが512以外はSLO対象外、debugはSLO対象外、ベクトルDBは同一マシン）を明記、debug取得p99<=500ms（SLO分離）を明記、レイテンシ計測スクリプトと生ログ保存方針を定義
- 成果物: 推論エンジン
- 完了条件: Pure Infer p50<=25ms/p99<=40ms、E2E p50<=45ms/p99<=80ms、同時4でE2E p99<=120ms

## 17. API実装とOpenAPI作成
- 目的: /infer /health /version /debug を提供
- 作業: スキーマ固定（schema_version/必須フィールド/enum）、必須フィールド（schema_version/inference_id/model_version/asof_timestamp_utc/latency_ms/per_horizon/historical_analogy/is_ood/ood_score/risk_flags）を明記、horizons_minは省略可（指定時は[10,30,60]のみ許容）、per_horizonは10/30/60昇順で一意かつ各要素に horizon_min を必須、range_forecast.targetは close_t_plus_{H}m 固定、debug I/F方式（GET /debug or debug_artifact_url）を選定して固定、debug=true時のみdebug_artifact_url（TTL=24h）、embedding_128は返却しない場合は省略（null/空配列不可）、market_condition.entropyの必須化、確率和=1.0（許容誤差±1e-6）、認証方式（mTLS or Bearer）を仕様に反映、400/401/403/413/422/429/500/503の実装（500は追跡ID付与）
- 成果物: APIサーバ、OpenAPI仕様
- 完了条件: 仕様通りレスポンスが返る（要件定義書v5.0 13章/16章準拠）

## 18. 監視・ログ設計
- 目的: 運用監視とデバッグの基盤
- 作業: latency(p50/p95/p99)/OOD率/data_integrity_warning率/確率・レンジドリフト、inference_id/asof/model_version/latency/ood_score/risk_flagsのログ化、予測結果（確率要約/entropy/range要約）のログ化、価格系列を保存する場合の暗号化・権限管理ポリシー、認証失敗・レート制限発火の監査ログを設計
- 成果物: メトリクス/ログ設計
- 完了条件: 主要メトリクスが収集できる（要件定義書v5.0 15章/16章準拠）

## 19. 評価スクリプト一式
- 目的: 要件定義書v5.0 第14章の合否判定を自動化
- 作業: Direction（Balanced Acc/Macro-F1/Recall、混同行列、期間別レポート）、Range（Pinball/coverage：P(y<=q10)=0.10±0.03、P(y<=q90)=0.90±0.03）、ECE、OOD(TPR/FPR)、Latency、類似検索Recall@10（教師信号・embargo・複数期間再現を含む）をホライズン別に評価、合否閾値（14.2/14.3/14.4/10.4/4.2）をコード内で固定、レイテンシ計測スクリプトと生ログ保存
- 成果物: 評価スクリプト、レポート、生ログ
- 完了条件: 合否が自動で判定される

## 20. Runbookと成果物整理
- 目的: 提出物・運用手順を完成
- 作業: Docker起動手順、モデルカード、ロールバック手順、OpenAPI仕様書とサンプル、計測スクリプトと評価レポート整理、SBOM作成方針の明記、秘密情報の扱い（Vault/環境変数、ハードコード禁止）の明記、OOD方式/閾値/再現手順の文書化
- 成果物: Runbook、成果物一覧
- 完了条件: 要件定義書v5.0 第17章の成果物が揃う
