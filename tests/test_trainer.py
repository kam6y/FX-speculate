"""model/trainer.py のテスト。"""

import pytest
from model.trainer import build_trainer, build_tft


class TestBuildTrainer:
    def test_returns_trainer(self):
        trainer = build_trainer(max_epochs=1, fast_dev_run=True)
        assert trainer is not None
        assert trainer.max_epochs == 1


class TestBuildTft:
    def test_returns_tft_with_correct_params(self):
        import pandas as pd
        import numpy as np
        from data.dataset import prepare_data, create_datasets

        n = 500
        index = pd.bdate_range("2024-01-02", periods=n)
        df = pd.DataFrame({
            "log_return": np.random.randn(n) * 0.01,
            "sma_5": np.random.randn(n),
            "day_of_week": index.dayofweek,
            "month": index.month,
            "is_month_end": index.is_month_end.astype(int),
            "days_to_next_major_event": np.random.randint(0, 20, n),
            "days_from_last_major_event": np.random.randint(0, 20, n),
            "event_type_next": np.random.choice(["FOMC", "NFP", "CPI"], n),
            "is_event_day": np.random.randint(0, 2, n),
            "event_density_past_5d": np.random.randint(0, 3, n),
        }, index=index)
        prepped = prepare_data(df)
        training, _ = create_datasets(prepped.iloc[:380], prepped.iloc[380:])
        tft = build_tft(training)
        assert tft is not None
