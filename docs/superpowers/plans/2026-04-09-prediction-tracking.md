# 予測 vs 実績トラッキング + 日次自動予測 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 過去の予測を実績と突き合わせて的中/外れを追跡するダッシュボードセクションを追加し、Windows タスクスケジューラで毎朝7:00に自動予測を実行する。

**Architecture:** `predict.py` に backfill 関数を追加し、日次実行時に過去予測の実績を DB に書き戻す。ダッシュボードは DB から実績付き予測を読んで表示する。自動実行は bat ファイル + schtasks で実現。

**Tech Stack:** Python, SQLite, Streamlit, Plotly, pandas, Windows Task Scheduler

---

## ファイル構成

| ファイル | 操作 | 責務 |
|---------|------|------|
| `scripts/predict.py` | 修正 | backfill ロジック追加、DB マイグレーション |
| `dashboard/app.py` | 修正 | 予測 vs 実績パネル追加 |
| `scripts/run_daily.bat` | 新規 | タスクスケジューラ用バッチファイル |
| `tests/test_predict.py` | 新規 | backfill ロジックのテスト |

---

### Task 1: backfill ロジックのテストを書く

**Files:**
- Create: `tests/test_predict.py`

- [ ] **Step 1: テストファイルを作成**

```python
"""predict.py の backfill ロジックのテスト。"""

import sqlite3
from datetime import date

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def predictions_db(tmp_path):
    """テスト用の predictions DB を作成する。"""
    db_path = tmp_path / "predictions.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE predictions (
            prediction_date TEXT,
            target_date TEXT,
            horizon INTEGER,
            median REAL,
            direction_signal REAL,
            q10 REAL,
            q90 REAL,
            threshold REAL,
            direction TEXT,
            actual_return REAL,
            actual_direction TEXT,
            is_correct INTEGER
        )
    """)
    # 2024-01-02 に予測、対象日 2024-01-03 (実績あり想定)
    conn.execute("""
        INSERT INTO predictions
        (prediction_date, target_date, horizon, median, direction_signal,
         q10, q90, threshold, direction)
        VALUES ('2024-01-02', '2024-01-03', 1, 0.001, 0.0012,
                -0.005, 0.007, 0.0, 'UP')
    """)
    # 未来日の予測 (実績なし想定)
    conn.execute("""
        INSERT INTO predictions
        (prediction_date, target_date, horizon, median, direction_signal,
         q10, q90, threshold, direction)
        VALUES ('2024-01-02', '2099-12-31', 2, -0.001, -0.0008,
                -0.006, 0.004, 0.0, 'DOWN')
    """)
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def price_series():
    """テスト用の USD/JPY close 価格 Series。"""
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    prices = pd.Series([141.0, 142.0, 141.5], index=dates, name="usdjpy_close")
    return prices


def test_backfill_updates_actual_return(predictions_db, price_series):
    """backfill が actual_return, actual_direction, is_correct を正しく埋める。"""
    from scripts.predict import backfill_actuals

    backfill_actuals(predictions_db, price_series)

    conn = sqlite3.connect(str(predictions_db))
    rows = conn.execute(
        "SELECT target_date, actual_return, actual_direction, is_correct "
        "FROM predictions ORDER BY target_date"
    ).fetchall()
    conn.close()

    # 2024-01-03: log(142/141) ≈ 0.00708 → UP → direction=UP → is_correct=1
    assert rows[0][0] == "2024-01-03"
    assert rows[0][1] == pytest.approx(np.log(142.0 / 141.0), abs=1e-6)
    assert rows[0][2] == "UP"
    assert rows[0][3] == 1

    # 2099-12-31: 価格データなし → NULL のまま
    assert rows[1][0] == "2099-12-31"
    assert rows[1][1] is None
    assert rows[1][2] is None
    assert rows[1][3] is None


def test_backfill_skips_already_filled(predictions_db, price_series):
    """既に actual_return が埋まっているレコードは更新しない。"""
    from scripts.predict import backfill_actuals

    conn = sqlite3.connect(str(predictions_db))
    conn.execute(
        "UPDATE predictions SET actual_return=0.999, actual_direction='UP', is_correct=0 "
        "WHERE target_date='2024-01-03'"
    )
    conn.commit()
    conn.close()

    backfill_actuals(predictions_db, price_series)

    conn = sqlite3.connect(str(predictions_db))
    row = conn.execute(
        "SELECT actual_return FROM predictions WHERE target_date='2024-01-03'"
    ).fetchone()
    conn.close()

    # 上書きされない
    assert row[0] == pytest.approx(0.999)


def test_backfill_down_direction(predictions_db, price_series):
    """実績が DOWN のケースで is_correct が正しく判定される。"""
    from scripts.predict import backfill_actuals

    # direction=UP だが実績は DOWN になるケースを作る
    conn = sqlite3.connect(str(predictions_db))
    conn.execute(
        "INSERT INTO predictions "
        "(prediction_date, target_date, horizon, median, direction_signal, "
        "q10, q90, threshold, direction) "
        "VALUES ('2024-01-03', '2024-01-04', 1, 0.001, 0.001, -0.005, 0.007, 0.0, 'UP')"
    )
    conn.commit()
    conn.close()

    backfill_actuals(predictions_db, price_series)

    conn = sqlite3.connect(str(predictions_db))
    row = conn.execute(
        "SELECT actual_return, actual_direction, is_correct "
        "FROM predictions WHERE prediction_date='2024-01-03' AND target_date='2024-01-04'"
    ).fetchone()
    conn.close()

    # log(141.5/142.0) < 0 → DOWN, direction=UP → is_correct=0
    assert row[0] == pytest.approx(np.log(141.5 / 142.0), abs=1e-6)
    assert row[1] == "DOWN"
    assert row[2] == 0
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `PYTHONPATH=. uv run python -m pytest tests/test_predict.py -v`
Expected: FAIL with `ImportError: cannot import name 'backfill_actuals'`

- [ ] **Step 3: コミット**

```bash
git add tests/test_predict.py
git commit -m "test: add backfill_actuals tests for prediction tracking"
```

---

### Task 2: backfill ロジックと DB マイグレーションを実装

**Files:**
- Modify: `scripts/predict.py`

- [ ] **Step 1: backfill_actuals 関数と migrate_db 関数を実装**

`scripts/predict.py` の先頭付近（import の後、`build_decoder_data` の前）に以下を追加:

```python
def migrate_db(db_path: Path) -> None:
    """predictions テーブルに実績カラムを追加する（冪等）。"""
    with sqlite3.connect(str(db_path)) as conn:
        cursor = conn.execute("PRAGMA table_info(predictions)")
        existing_cols = {row[1] for row in cursor.fetchall()}
        for col, col_type in [
            ("actual_return", "REAL"),
            ("actual_direction", "TEXT"),
            ("is_correct", "INTEGER"),
        ]:
            if col not in existing_cols:
                conn.execute(f"ALTER TABLE predictions ADD COLUMN {col} {col_type}")


def backfill_actuals(db_path: Path, price_series: pd.Series) -> None:
    """過去の予測に実績データを書き戻す。

    Args:
        db_path: predictions.db のパス
        price_series: USD/JPY close 価格の Series (index=日付)
    """
    with sqlite3.connect(str(db_path)) as conn:
        pending = pd.read_sql(
            "SELECT rowid, prediction_date, target_date, direction "
            "FROM predictions "
            "WHERE actual_return IS NULL AND target_date <= ?",
            conn,
            params=(str(date.today()),),
        )

    if pending.empty:
        return

    price_index = price_series.index.normalize()
    updates = []

    for _, row in pending.iterrows():
        pred_date = pd.Timestamp(row["prediction_date"])
        tgt_date = pd.Timestamp(row["target_date"])

        # 両方の日付の close が必要
        pred_mask = price_index == pred_date
        tgt_mask = price_index == tgt_date

        if not pred_mask.any() or not tgt_mask.any():
            continue

        close_pred = float(price_series[pred_mask].iloc[0])
        close_tgt = float(price_series[tgt_mask].iloc[0])

        actual_return = float(np.log(close_tgt / close_pred))
        actual_direction = "UP" if actual_return > 0 else "DOWN"
        is_correct = 1 if row["direction"] == actual_direction else 0

        updates.append((actual_return, actual_direction, is_correct, row["rowid"]))

    if updates:
        with sqlite3.connect(str(db_path)) as conn:
            conn.executemany(
                "UPDATE predictions SET actual_return=?, actual_direction=?, is_correct=? "
                "WHERE rowid=?",
                updates,
            )
        print(f"  Backfilled {len(updates)} predictions with actuals")
```

`predict.py` の import セクションに `Path` を追加（既に `from pathlib import Path` 相当が `config` 経由であるが明示的に不要。`date` は既に import 済み）。

- [ ] **Step 2: predict_daily() に backfill 呼び出しを組み込む**

`predict_daily()` 関数の先頭（`print("=== 最新データ取得 ===")` の直前）に以下を追加:

```python
    # DB マイグレーション（新カラム追加）
    if PREDICTIONS_DB.exists():
        migrate_db(PREDICTIONS_DB)

    # 過去予測の実績を書き戻す
    print("=== 実績データ Backfill ===")
```

`raw = fetch_all_data(use_cache=False)` の直後に以下を追加:

```python
    # backfill (raw データ取得後に実行)
    if PREDICTIONS_DB.exists():
        backfill_actuals(PREDICTIONS_DB, raw["usdjpy_close"])
```

- [ ] **Step 3: テストが通ることを確認**

Run: `PYTHONPATH=. uv run python -m pytest tests/test_predict.py -v`
Expected: 3 tests PASS

- [ ] **Step 4: 既存テストが壊れていないことを確認**

Run: `PYTHONPATH=. uv run python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 5: コミット**

```bash
git add scripts/predict.py
git commit -m "feat: add backfill_actuals to write actual returns into predictions DB"
```

---

### Task 3: ダッシュボードに予測 vs 実績パネルを追加

**Files:**
- Modify: `dashboard/app.py`

- [ ] **Step 1: load_predictions の返すデータに実績カラムが含まれることを確認**

`load_predictions()` は `SELECT * FROM predictions` を実行しているので、新カラム（`actual_return`, `actual_direction`, `is_correct`）は自動的に含まれる。変更不要。

- [ ] **Step 2: panel_prediction_tracking 関数を追加**

`dashboard/app.py` の `panel_accuracy_history` 関数の直前に以下を追加:

```python
def panel_prediction_tracking(preds: pd.DataFrame) -> None:
    """予測 vs 実績: 過去の予測の的中/外れを追跡。"""
    st.subheader("予測 vs 実績トラッキング")
    st.caption(
        "過去の予測に対して実際の値動きがどうだったかを追跡。"
        "的中率が 50% (ランダム基準) を安定的に上回っているかが重要。"
    )

    if preds.empty or "is_correct" not in preds.columns:
        st.info("データがありません。scripts/predict.py を実行してください。")
        return

    filled = preds[preds["is_correct"].notna()].copy()
    pending = preds[preds["is_correct"].isna()]

    if filled.empty:
        st.info(
            f"実績データがまだありません（{len(pending)} 件が待機中）。"
            "対象日を過ぎた後に predict.py を再実行すると実績が反映されます。"
        )
        return

    # --- 的中率サマリー ---
    overall_acc = filled["is_correct"].mean()
    st.metric("全体的中率", f"{overall_acc:.1%}", delta=f"{overall_acc - 0.5:+.1%} vs ランダム")

    cols = st.columns(PREDICTION_LENGTH)
    for h in range(1, PREDICTION_LENGTH + 1):
        h_data = filled[filled["horizon"] == h]
        if h_data.empty:
            cols[h - 1].metric(f"{h}日後", "N/A")
        else:
            acc = h_data["is_correct"].mean()
            n = len(h_data)
            cols[h - 1].metric(f"{h}日後", f"{acc:.1%}", delta=f"n={n}")

    # --- 予測履歴テーブル ---
    st.markdown("#### 予測履歴")

    display = preds.sort_values(
        ["prediction_date", "horizon"], ascending=[False, True]
    ).copy()
    display["結果"] = display["is_correct"].map({1.0: "○", 0.0: "×"}).fillna("待機中")
    display["実績方向"] = display["actual_direction"].fillna("-")

    table_df = display.rename(columns={
        "prediction_date": "予測日",
        "target_date": "対象日",
        "horizon": "H",
        "direction": "予測方向",
    })[["予測日", "対象日", "H", "予測方向", "実績方向", "結果"]]

    def highlight_result(row):
        if row["結果"] == "○":
            return ["background-color: rgba(0, 204, 150, 0.2)"] * len(row)
        elif row["結果"] == "×":
            return ["background-color: rgba(239, 85, 59, 0.2)"] * len(row)
        return [""] * len(row)

    styled = table_df.style.apply(highlight_result, axis=1)
    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        height=min(35 * (len(table_df) + 1) + 10, 500),
    )

    # --- 的中率推移グラフ ---
    if len(filled["prediction_date"].unique()) >= 2:
        st.markdown("#### 的中率推移")
        by_date = (
            filled.groupby("prediction_date")["is_correct"]
            .mean()
            .reset_index()
            .sort_values("prediction_date")
        )

        fig = go.Figure(go.Scatter(
            x=by_date["prediction_date"],
            y=by_date["is_correct"],
            mode="lines+markers",
            line=dict(color="#636EFA", width=2),
            marker=dict(size=8),
            name="的中率",
        ))
        fig.add_hline(
            y=0.5, line_dash="dash", line_color="gray",
            annotation_text="ランダム基準 (50%)",
        )
        fig.update_layout(
            yaxis=dict(title="的中率", range=[0, 1], tickformat=".0%"),
            xaxis_title="予測日",
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)
```

- [ ] **Step 3: main() に新パネルを配置**

`dashboard/app.py` の `main()` 関数内、`panel_direction_signals(preds_df)` の直後の `st.divider()` のあとに追加:

既存コード:
```python
    panel_direction_signals(preds_df)
    st.divider()

    # --- 中段: 方向比率 + イベントカレンダー ---
```

変更後:
```python
    panel_direction_signals(preds_df)
    st.divider()

    # --- 予測 vs 実績 ---
    panel_prediction_tracking(preds_df)
    st.divider()

    # --- 中段: 方向比率 + イベントカレンダー ---
```

- [ ] **Step 4: ダッシュボードが起動することを確認**

Run: `PYTHONPATH=. uv run streamlit run dashboard/app.py --server.headless true` (Ctrl+C で停止)
Expected: エラーなく起動する

- [ ] **Step 5: コミット**

```bash
git add dashboard/app.py
git commit -m "feat: add prediction vs actual tracking panel to dashboard"
```

---

### Task 4: 日次自動実行バッチファイルを作成

**Files:**
- Create: `scripts/run_daily.bat`

- [ ] **Step 1: バッチファイルを作成**

```bat
@echo off
REM USD/JPY TFT 日次予測 - タスクスケジューラ用
REM 毎朝 7:00 に実行される想定

cd /d C:\Users\daiya\Documents\FX-speculate

set PYTHONPATH=.

echo [%date% %time%] Starting daily prediction... >> logs\daily_predict.log
uv run python scripts/predict.py >> logs\daily_predict.log 2>&1
echo [%date% %time%] Done (exit code: %ERRORLEVEL%) >> logs\daily_predict.log
```

- [ ] **Step 2: logs ディレクトリを .gitignore に追加**

`.gitignore` に以下を追加:

```
logs/
```

- [ ] **Step 3: コミット**

```bash
git add scripts/run_daily.bat .gitignore
git commit -m "feat: add daily prediction batch file for Task Scheduler"
```

---

### Task 5: Windows タスクスケジューラに登録

**Files:** なし（OS 設定のみ）

- [ ] **Step 1: logs ディレクトリを作成**

```bash
mkdir -p logs
```

- [ ] **Step 2: タスクスケジューラに登録**

管理者権限のコマンドプロンプトで実行:

```cmd
schtasks /create /tn "FX-Speculate Daily Predict" /tr "C:\Users\daiya\Documents\FX-speculate\scripts\run_daily.bat" /sc daily /st 07:00 /f
```

- [ ] **Step 3: 登録確認**

```cmd
schtasks /query /tn "FX-Speculate Daily Predict" /v
```

Expected: タスクが表示され、次回実行時刻が翌朝7:00になっている

- [ ] **Step 4: 手動テスト実行**

```cmd
schtasks /run /tn "FX-Speculate Daily Predict"
```

`logs/daily_predict.log` に出力が書き込まれることを確認。

---

### Task 6: 既存 predictions DB をマイグレーション + 動作確認

**Files:** なし（手動実行）

- [ ] **Step 1: predict.py を手動実行して DB マイグレーション + backfill を確認**

```bash
PYTHONPATH=. uv run python scripts/predict.py
```

Expected:
- `=== 実績データ Backfill ===` が表示される
- `Backfilled N predictions with actuals` が表示される（過去予測の対象日が到来している場合）
- 新しい予測が DB に追加される

- [ ] **Step 2: DB の中身を確認**

```bash
uv run python -c "
import sqlite3
conn = sqlite3.connect('artifacts/predictions.db')
rows = conn.execute('SELECT prediction_date, target_date, direction, actual_direction, is_correct FROM predictions').fetchall()
for r in rows:
    print(r)
conn.close()
"
```

Expected: 過去の予測に `actual_direction` と `is_correct` が埋まっている

- [ ] **Step 3: ダッシュボードで確認**

```bash
PYTHONPATH=. uv run streamlit run dashboard/app.py
```

Expected: 「予測 vs 実績トラッキング」セクションに的中率と履歴テーブルが表示される

- [ ] **Step 4: 全テスト実行**

```bash
PYTHONPATH=. uv run python -m pytest tests/ -v
```

Expected: All tests PASS

- [ ] **Step 5: 最終コミット**

```bash
git add -A
git commit -m "chore: verify prediction tracking end-to-end"
```
