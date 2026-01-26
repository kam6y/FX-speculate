import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from pms.imputation import fill_ohlcv_gaps  # noqa: E402
from pms.validation import OhlcvBar, RiskFlags, ValidationError  # noqa: E402


def make_bar(timestamp_utc: datetime, close_value: float, volume: float = 1.0) -> OhlcvBar:
    return OhlcvBar(
        timestamp_utc=timestamp_utc,
        open=close_value,
        high=close_value,
        low=close_value,
        close=close_value,
        volume=volume,
    )


class ImputationTests(unittest.TestCase):
    def test_short_gap_fill_adds_reason_and_imputes(self) -> None:
        start = datetime(2026, 1, 26, 0, 0, tzinfo=timezone.utc)
        bars = [
            make_bar(start + timedelta(minutes=0), 100.0),
            make_bar(start + timedelta(minutes=1), 101.0),
            make_bar(start + timedelta(minutes=2), 102.0),
            make_bar(start + timedelta(minutes=5), 105.0),
            make_bar(start + timedelta(minutes=6), 106.0),
        ]

        risk_flags = RiskFlags()
        filled = fill_ohlcv_gaps(bars, lookback_bars=5, timeframe_sec=60, risk_flags=risk_flags)

        expected_times = [start + timedelta(minutes=m) for m in [2, 3, 4, 5, 6]]
        self.assertEqual([bar.timestamp_utc for bar in filled], expected_times)
        self.assertEqual(filled[1].close, 102.0)
        self.assertEqual(filled[2].close, 102.0)
        self.assertEqual(filled[1].volume, 0.0)
        self.assertIn("ohlcv_gap_filled", risk_flags.degraded_reasons)
        self.assertTrue(risk_flags.data_integrity_warning)
        self.assertTrue(risk_flags.degraded)

    def test_long_gap_adds_too_long_reason(self) -> None:
        start = datetime(2026, 1, 26, 1, 0, tzinfo=timezone.utc)
        bars = [
            make_bar(start + timedelta(minutes=0), 200.0),
            make_bar(start + timedelta(minutes=1), 201.0),
            make_bar(start + timedelta(minutes=2), 202.0),
            make_bar(start + timedelta(minutes=7), 207.0),
            make_bar(start + timedelta(minutes=8), 208.0),
        ]

        risk_flags = RiskFlags()
        filled = fill_ohlcv_gaps(bars, lookback_bars=5, timeframe_sec=60, risk_flags=risk_flags)

        self.assertEqual(len(filled), 5)
        self.assertIn("ohlcv_gap_too_long", risk_flags.degraded_reasons)
        self.assertNotIn("ohlcv_gap_filled", risk_flags.degraded_reasons)

    def test_gap_exceeds_max_raises(self) -> None:
        start = datetime(2026, 1, 26, 2, 0, tzinfo=timezone.utc)
        bars = [
            make_bar(start, 300.0),
            make_bar(start + timedelta(minutes=62), 301.0),
        ]

        with self.assertRaises(ValidationError) as ctx:
            fill_ohlcv_gaps(bars, lookback_bars=2, timeframe_sec=60, risk_flags=RiskFlags())
        self.assertEqual(ctx.exception.status_code, 422)

    def test_weekly_close_gap_is_ignored(self) -> None:
        friday = datetime(2026, 1, 30, 20, 59, tzinfo=timezone.utc)
        sunday = datetime(2026, 2, 1, 22, 0, tzinfo=timezone.utc)
        bars = [
            make_bar(friday, 150.0),
            make_bar(sunday, 151.0),
        ]

        risk_flags = RiskFlags()
        filled = fill_ohlcv_gaps(bars, lookback_bars=2, timeframe_sec=60, risk_flags=risk_flags)

        self.assertEqual([bar.timestamp_utc for bar in filled], [friday, sunday])
        self.assertFalse(risk_flags.degraded)
        self.assertFalse(risk_flags.data_integrity_warning)

    def test_volume_missing_sets_reason(self) -> None:
        start = datetime(2026, 1, 26, 3, 0, tzinfo=timezone.utc)
        bars = [
            make_bar(start, 400.0, volume=0.0),
            make_bar(start + timedelta(minutes=1), 401.0, volume=1.0),
        ]

        risk_flags = RiskFlags()
        fill_ohlcv_gaps(bars, lookback_bars=2, timeframe_sec=60, risk_flags=risk_flags)

        self.assertIn("volume_missing", risk_flags.degraded_reasons)
        self.assertTrue(risk_flags.data_integrity_warning)
        self.assertTrue(risk_flags.degraded)


if __name__ == "__main__":
    unittest.main()
