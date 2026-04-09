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
