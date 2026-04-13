# 予測 vs 実績トラッキング + 日次自動予測

## 概要

1. 過去の予測結果に実績データを紐付けて的中/外れを追跡するダッシュボードセクション
2. Windows タスクスケジューラによる毎朝7:00の自動予測実行

## 1. DBスキーマ変更

`predictions` テーブルに3カラムを `ALTER TABLE` で追加:

```sql
ALTER TABLE predictions ADD COLUMN actual_return REAL;
ALTER TABLE predictions ADD COLUMN actual_direction TEXT;
ALTER TABLE predictions ADD COLUMN is_correct INTEGER;
```

nullable のまま。backfill 時に値を入れる。

## 2. predict.py: backfill ロジック

`predict_daily()` の冒頭で以下を実行:

1. DBから `actual_return IS NULL AND target_date <= today` のレコードを取得
2. `fetch_all_data()` で最新の USD/JPY close を取得（predict 本体でも使うので1回で済む）
3. 各レコードについて:
   - `prediction_date` と `target_date` の close 価格を取得
   - `actual_return = log(close[target_date] / close[prediction_date])`
   - `actual_direction = "UP" if actual_return > 0 else "DOWN"`
   - `is_correct = 1 if direction == actual_direction else 0`
4. DB を UPDATE

close 価格が取得できない日（祝日等）はスキップし、次回に持ち越す。

## 3. ダッシュボード: 予測 vs 実績パネル

`dashboard/app.py` に `panel_prediction_tracking()` を追加。

### 3.1 的中率サマリー

`st.metric` × 5（ホライゾン別）+ 全体の的中率を上部に表示。

### 3.2 予測履歴テーブル

columns: 予測日, 対象日, ホライゾン, 予測方向, 実績方向, 結果(○/×)
- is_correct=1 の行は緑背景、0 は赤背景
- actual_return IS NULL の行は「待機中」表示
- prediction_date の降順でソート

### 3.3 的中率推移グラフ

prediction_date ごとに5ホライゾンの平均的中率を算出し、折れ線グラフで推移を表示。
50% のランダム基準ラインを破線で表示。

### レイアウト配置

既存の「方向比率モニター + イベントカレンダー」の上に新セクションとして挿入:

```
予測チャート
---
方向シグナル
---
★ 予測 vs 実績 (NEW)
---
方向比率 | イベントカレンダー
---
特徴量重要度 | Attention
---
過去予測の精度
```

## 4. 日次自動実行

### 4.1 バッチファイル

`scripts/run_daily.bat`:
- プロジェクトディレクトリへ cd
- PYTHONPATH を設定
- `uv run python scripts/predict.py` を実行
- ログをファイルに出力

### 4.2 タスクスケジューラ登録

`schtasks /create` で毎日7:00に実行するタスクを登録。

## 5. 変更ファイル一覧

| ファイル | 変更 |
|---------|------|
| `scripts/predict.py` | backfill ロジック追加、DB マイグレーション |
| `dashboard/app.py` | `panel_prediction_tracking()` 追加 |
| `scripts/run_daily.bat` | 新規作成 |
