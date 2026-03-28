"""SQLite データベース管理モジュール

テーブル:
  - predictions: 日次予測結果
  - model_metrics: モデル評価メトリクス (学習時スナップショット)
  - alerts: 運用アラート (Section 7.3)
"""

import sqlite3
from datetime import datetime, date
from pathlib import Path
from contextlib import contextmanager

DB_PATH = Path(__file__).parent / "tft_dashboard.db"


@contextmanager
def get_conn():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """テーブル作成 (冪等)"""
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_date TEXT NOT NULL,
                target_date TEXT NOT NULL,
                horizon INTEGER NOT NULL DEFAULT 1,
                pred_log_return REAL NOT NULL,
                pred_q10 REAL,
                pred_q90 REAL,
                direction TEXT NOT NULL,
                confidence REAL,
                actual_log_return REAL,
                actual_direction TEXT,
                is_correct INTEGER,
                current_price REAL,
                created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
                UNIQUE(prediction_date, target_date, horizon)
            );

            CREATE TABLE IF NOT EXISTS model_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recorded_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                source TEXT DEFAULT 'evaluation'
            );

            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL DEFAULT 'info',
                message TEXT NOT NULL,
                is_resolved INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
                resolved_at TEXT
            );

            CREATE TABLE IF NOT EXISTS daily_pnl (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_date TEXT NOT NULL UNIQUE,
                signal TEXT NOT NULL,
                pred_log_return REAL NOT NULL,
                actual_log_return REAL,
                pnl REAL,
                cumulative_pnl REAL,
                created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
            );
        """)


# ─── Predictions ───

def save_prediction(prediction_date: str, target_date: str, horizon: int,
                    pred_lr: float, pred_q10: float, pred_q90: float,
                    direction: str, confidence: float, current_price: float):
    with get_conn() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO predictions
                (prediction_date, target_date, horizon, pred_log_return,
                 pred_q10, pred_q90, direction, confidence, current_price)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (prediction_date, target_date, horizon, pred_lr,
              pred_q10, pred_q90, direction, confidence, current_price))


def update_actual(target_date: str, horizon: int,
                  actual_lr: float, actual_direction: str):
    is_correct = None
    with get_conn() as conn:
        row = conn.execute(
            "SELECT direction FROM predictions WHERE target_date=? AND horizon=?",
            (target_date, horizon)
        ).fetchone()
        if row:
            is_correct = 1 if row["direction"] == actual_direction else 0
        conn.execute("""
            UPDATE predictions
            SET actual_log_return=?, actual_direction=?, is_correct=?
            WHERE target_date=? AND horizon=?
        """, (actual_lr, actual_direction, is_correct, target_date, horizon))


def get_recent_predictions(limit: int = 30) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT * FROM predictions
            WHERE horizon = 1
            ORDER BY target_date DESC
            LIMIT ?
        """, (limit,)).fetchall()
        return [dict(r) for r in rows]


def get_latest_prediction() -> dict | None:
    with get_conn() as conn:
        row = conn.execute("""
            SELECT * FROM predictions
            WHERE horizon = 1
            ORDER BY target_date DESC
            LIMIT 1
        """).fetchone()
        return dict(row) if row else None


def get_multi_horizon_prediction(prediction_date: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT * FROM predictions
            WHERE prediction_date = ?
            ORDER BY horizon
        """, (prediction_date,)).fetchall()
        return [dict(r) for r in rows]


# ─── Model Metrics ───

def save_metrics(metrics: dict, source: str = "evaluation"):
    with get_conn() as conn:
        now = datetime.now().isoformat()
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                conn.execute("""
                    INSERT INTO model_metrics (recorded_at, metric_name, metric_value, source)
                    VALUES (?, ?, ?, ?)
                """, (now, name, value, source))


def get_latest_metrics() -> dict:
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT metric_name, metric_value
            FROM model_metrics
            WHERE recorded_at = (SELECT MAX(recorded_at) FROM model_metrics)
        """).fetchall()
        return {r["metric_name"]: r["metric_value"] for r in rows}


# ─── Alerts ───

def create_alert(alert_type: str, severity: str, message: str):
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO alerts (alert_type, severity, message)
            VALUES (?, ?, ?)
        """, (alert_type, severity, message))


def get_active_alerts() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT * FROM alerts
            WHERE is_resolved = 0
            ORDER BY created_at DESC
        """).fetchall()
        return [dict(r) for r in rows]


def resolve_alert(alert_id: int):
    with get_conn() as conn:
        conn.execute("""
            UPDATE alerts SET is_resolved = 1, resolved_at = datetime('now', 'localtime')
            WHERE id = ?
        """, (alert_id,))


# ─── Daily PnL ───

def save_daily_pnl(trade_date: str, signal: str, pred_lr: float,
                   actual_lr: float | None, pnl: float | None,
                   cumulative_pnl: float | None):
    with get_conn() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO daily_pnl
                (trade_date, signal, pred_log_return, actual_log_return, pnl, cumulative_pnl)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (trade_date, signal, pred_lr, actual_lr, pnl, cumulative_pnl))


def get_pnl_history(limit: int = 500) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT * FROM daily_pnl
            ORDER BY trade_date ASC
            LIMIT ?
        """, (limit,)).fetchall()
        return [dict(r) for r in rows]


def get_rolling_accuracy(window: int = 60) -> list[dict]:
    """直近の方向精度をローリング計算"""
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT target_date, is_correct
            FROM predictions
            WHERE horizon = 1 AND is_correct IS NOT NULL
            ORDER BY target_date ASC
        """).fetchall()

    if not rows:
        return []

    results = []
    data = [dict(r) for r in rows]
    for i in range(len(data)):
        start = max(0, i - window + 1)
        window_data = data[start:i + 1]
        correct = sum(1 for d in window_data if d["is_correct"] == 1)
        acc = correct / len(window_data) if window_data else 0
        results.append({
            "date": data[i]["target_date"],
            "accuracy": round(acc, 4),
            "window_size": len(window_data),
        })
    return results


# ─── Live Equity (DB蓄積予測から計算) ───

def get_predictions_with_actuals() -> list[dict]:
    """実績値が入った1D予測を日付順で取得"""
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT target_date, direction, pred_log_return,
                   actual_log_return, actual_direction, is_correct
            FROM predictions
            WHERE horizon = 1 AND actual_log_return IS NOT NULL
            ORDER BY target_date ASC
        """).fetchall()
        return [dict(r) for r in rows]


def get_pending_target_dates() -> list[str]:
    """実績値が未入力の予測対象日を取得"""
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT DISTINCT target_date FROM predictions
            WHERE horizon = 1 AND actual_log_return IS NULL
            ORDER BY target_date ASC
        """).fetchall()
        return [r["target_date"] for r in rows]
