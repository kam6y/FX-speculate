# PMS Feature Dictionary v5.0（USDJPY）

- 文書ID：PMS-FEATURE-DICT-USDJPY-v5.0
- 作成日：2026-01-26
- 対象：USD/JPY

本書は PMS-REQ-USDJPY-v5.0 の 7章（特徴量要件）に基づき、特徴量名・定義・生成規則を固定する。
WBS Step 5「特徴量辞書と生成モジュール作成」の成果物として扱う。

## 1. 参照元（ソースオブトゥルース）
- PMS-REQ-USDJPY-v5.0.md
  - 6.3（時刻特徴）
  - 7.1（必須特徴量）
  - 7.2（ロバスト性要件）
- PMS-REQ-USDJPY-v5.0-WBS.md
  - 5. 特徴量辞書と生成モジュール作成

## 2. 用語
- t：直近確定バー（asof_timestamp_utc - timeframe_sec）
- lookback_bars：推論に入力するOHLCV本数
- asof_timestamp_utc：推論基準時刻（バー確定時刻）

## 3. 生成ルール（固定）

### 3.1 価格系
- return_1：r_t = ln(close_t / close_{t-1})
- hl_range_norm：(high_t - low_t) / close_t

### 3.2 出来高系
- volume_log：ln(volume_t + 1)
- volume_z：z_t = (volume_t - median) / (MAD * 1.4826)
  - median/MAD は推論時に lookback_bars 範囲（t-lookback_bars+1..t）で算出
  - volume_missing_flag=1 の場合は volume_log=0, volume_z=0 に固定

### 3.3 ボラ系
- rv_30：sqrt( Σ_{i=t-29..t} r_i^2 )

### 3.4 時刻埋め込み（UTC）
- day_sin, day_cos（t の timestamp_utc を基準）
- week_sin, week_cos（週開始は UTC 月曜 00:00）

### 3.5 マクロ近傍
- event_onehot_{event_type}：
  - asof_timestamp_utc 以降 24h 以内で最も近い scheduled_time_utc を持つイベントを one-hot 化
  - 該当なしは全0
  - 同時刻タイブレーク：importance（high>medium>low）→ event_type 辞書順
- event_decay_past：
  - published_at_utc <= asof_timestamp_utc の最新イベントとの差分分数 delta_t から exp(-delta_t/tau) を算出
  - tau=60分、該当なしは0
- event_time_to_next：
  - 次回 scheduled_time_utc までの分数
  - 0〜1440分にクリップ、該当なしは1440

### 3.6 欠損マスク
- volume_missing_flag：volume_t == 0 の場合 1、それ以外 0

## 4. 特徴量一覧（最小セット）

|区分|特徴量名|型|定義/補足|
|---|---|---|---|
|価格|return_1|float|ln(close_t / close_{t-1})|
|価格|hl_range_norm|float|(high_t - low_t) / close_t|
|ボラ|rv_30|float|sqrt(sum r_i^2), i=t-29..t|
|出来高|volume_log|float|ln(volume_t + 1)、欠損時は0|
|出来高|volume_z|float|median/MAD*1.4826、欠損時は0|
|時刻|day_sin|float|UTC基準の1日周期sin|
|時刻|day_cos|float|UTC基準の1日周期cos|
|時刻|week_sin|float|UTC基準の1週周期sin|
|時刻|week_cos|float|UTC基準の1週周期cos|
|マクロ|event_onehot_{event_type}|float|辞書順でone-hot|
|マクロ|event_decay_past|float|exp(-delta_t/60)|
|マクロ|event_time_to_next|float|0〜1440分にクリップ|
|欠損|volume_missing_flag|float|volume_t==0で1|

## 5. ロバスト正規化方式（固定）
- 方式B（median/MAD）
- scale = MAD * 1.4826

## 6. 実装の固定パラメータ
- RV 窓長：30
- event_onehot 参照窓：asof_timestamp_utc 以降 24h
- event_decay_past：tau=60分
- event_time_to_next：0〜1440分にクリップ（該当なしは1440）
- event_type の辞書順：ASCII 昇順
- time embedding 基準：t（直近確定バー）
- week_sin/cos の週開始：UTC 月曜 00:00
- MAD=0 の場合：volume_z は 0 固定

## 7. 未確定事項（要確認）
- event_type の列挙リスト（PMS-REQ-USDJPY-v5.0 6.2）
