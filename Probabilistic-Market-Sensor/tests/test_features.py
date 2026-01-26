import math
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from pms.features import (  # noqa: E402
    FeatureConfig,
    FeatureError,
    build_feature_names,
    generate_feature_vector,
)
from pms.validation import MacroEvent, OhlcvBar  # noqa: E402


def make_bar(timestamp_utc: datetime, close_value: float, volume: float = 1.0) -> OhlcvBar:
    return OhlcvBar(
        timestamp_utc=timestamp_utc,
        open=close_value,
        high=close_value * 1.01,
        low=close_value * 0.99,
        close=close_value,
        volume=volume,
    )


class FeatureTests(unittest.TestCase):
    def test_feature_names_include_event_vocab_order(self) -> None:
        config = FeatureConfig(event_type_vocab=["CPI_US", "FOMC_RATE_DECISION"])
        names = build_feature_names(config)
        cpi_idx = names.index("event_onehot_CPI_US")
        fomc_idx = names.index("event_onehot_FOMC_RATE_DECISION")
        self.assertLess(cpi_idx, fomc_idx)

    def test_log_return_hl_range_and_rv(self) -> None:
        start = datetime(2026, 1, 26, 0, 0, tzinfo=timezone.utc)
        k = 0.01
        bars = []
        for idx in range(31):
            close_value = 100.0 * math.exp(k * idx)
            bars.append(make_bar(start + timedelta(minutes=idx), close_value, volume=100.0))

        asof = start + timedelta(minutes=31)
        config = FeatureConfig(event_type_vocab=["CPI_US"])
        result = generate_feature_vector(bars, [], asof, config, lookback_bars=len(bars))

        expected_return = k
        expected_hl_range = 0.02
        expected_rv = abs(k) * math.sqrt(30)

        self.assertAlmostEqual(result.feature_map["return_1"], expected_return, places=10)
        self.assertAlmostEqual(result.feature_map["hl_range_norm"], expected_hl_range, places=10)
        self.assertAlmostEqual(result.feature_map["rv_30"], expected_rv, places=10)

    def test_volume_missing_sets_zero_and_flag(self) -> None:
        start = datetime(2026, 1, 26, 1, 0, tzinfo=timezone.utc)
        bars = []
        for idx in range(31):
            volume = 10.0
            if idx == 30:
                volume = 0.0
            bars.append(make_bar(start + timedelta(minutes=idx), 100.0, volume=volume))

        asof = start + timedelta(minutes=31)
        config = FeatureConfig(event_type_vocab=["CPI_US"])
        result = generate_feature_vector(bars, [], asof, config, lookback_bars=len(bars))

        self.assertEqual(result.feature_map["volume_missing_flag"], 1.0)
        self.assertEqual(result.feature_map["volume_log"], 0.0)
        self.assertEqual(result.feature_map["volume_z"], 0.0)

    def test_volume_zscore_robust_median_mad(self) -> None:
        start = datetime(2026, 1, 26, 2, 0, tzinfo=timezone.utc)
        bars = []
        for idx in range(31):
            volume = float(idx + 1)
            bars.append(make_bar(start + timedelta(minutes=idx), 100.0, volume=volume))

        asof = start + timedelta(minutes=31)
        config = FeatureConfig(event_type_vocab=["CPI_US"])
        result = generate_feature_vector(bars, [], asof, config, lookback_bars=len(bars))

        median = 16.0
        mad = 8.0
        expected_z = (31.0 - median) / (mad * 1.4826)
        self.assertAlmostEqual(result.feature_map["volume_z"], expected_z, places=6)
        self.assertAlmostEqual(result.feature_map["volume_log"], math.log(32.0), places=10)
        self.assertEqual(result.feature_map["volume_missing_flag"], 0.0)

    def test_event_features_selection_and_decay(self) -> None:
        start = datetime(2026, 1, 26, 3, 0, tzinfo=timezone.utc)
        bars = [make_bar(start + timedelta(minutes=idx), 100.0, volume=10.0) for idx in range(31)]
        asof = start + timedelta(minutes=31)

        scheduled = asof + timedelta(minutes=60)
        published_latest = asof - timedelta(minutes=30)
        published_older = asof - timedelta(minutes=90)
        events = [
            MacroEvent(
                event_type="CPI_US",
                scheduled_time_utc=scheduled,
                importance="medium",
                revision_policy="none",
            ),
            MacroEvent(
                event_type="FOMC_RATE_DECISION",
                scheduled_time_utc=scheduled,
                importance="high",
                revision_policy="none",
            ),
            MacroEvent(
                event_type="CPI_US",
                scheduled_time_utc=published_older,
                importance="low",
                revision_policy="none",
                published_at_utc=published_older,
            ),
            MacroEvent(
                event_type="FOMC_RATE_DECISION",
                scheduled_time_utc=published_latest,
                importance="high",
                revision_policy="none",
                published_at_utc=published_latest,
            ),
        ]

        config = FeatureConfig(event_type_vocab=["CPI_US", "FOMC_RATE_DECISION"])
        result = generate_feature_vector(bars, events, asof, config, lookback_bars=len(bars))

        self.assertEqual(result.feature_map["event_onehot_FOMC_RATE_DECISION"], 1.0)
        self.assertEqual(result.feature_map["event_onehot_CPI_US"], 0.0)
        self.assertAlmostEqual(result.feature_map["event_time_to_next"], 60.0, places=6)
        self.assertAlmostEqual(
            result.feature_map["event_decay_past"],
            math.exp(-0.5),
            places=6,
        )

    def test_event_onehot_dictionary_order_tiebreak(self) -> None:
        start = datetime(2026, 1, 26, 4, 0, tzinfo=timezone.utc)
        bars = [make_bar(start + timedelta(minutes=idx), 100.0, volume=10.0) for idx in range(31)]
        asof = start + timedelta(minutes=31)
        scheduled = asof + timedelta(minutes=10)
        events = [
            MacroEvent(
                event_type="FOMC_RATE_DECISION",
                scheduled_time_utc=scheduled,
                importance="high",
                revision_policy="none",
            ),
            MacroEvent(
                event_type="CPI_US",
                scheduled_time_utc=scheduled,
                importance="high",
                revision_policy="none",
            ),
        ]

        config = FeatureConfig(event_type_vocab=["CPI_US", "FOMC_RATE_DECISION"])
        result = generate_feature_vector(bars, events, asof, config, lookback_bars=len(bars))

        self.assertEqual(result.feature_map["event_onehot_CPI_US"], 1.0)
        self.assertEqual(result.feature_map["event_onehot_FOMC_RATE_DECISION"], 0.0)

    def test_invalid_event_vocab_raises(self) -> None:
        with self.assertRaises(FeatureError):
            FeatureConfig(event_type_vocab=[])

    def test_event_vocab_requires_ascii_sorted(self) -> None:
        with self.assertRaises(FeatureError):
            FeatureConfig(event_type_vocab=["FOMC_RATE_DECISION", "CPI_US"])

    def test_volume_zscore_uses_lookback_window(self) -> None:
        start = datetime(2026, 1, 26, 5, 0, tzinfo=timezone.utc)
        bars = []
        for idx in range(40):
            volume = float(idx + 1)
            bars.append(make_bar(start + timedelta(minutes=idx), 100.0, volume=volume))

        asof = start + timedelta(minutes=40)
        config = FeatureConfig(event_type_vocab=["CPI_US"])
        result = generate_feature_vector(bars, [], asof, config, lookback_bars=31)

        expected_z = (40.0 - 25.0) / (8.0 * 1.4826)
        self.assertAlmostEqual(result.feature_map["volume_z"], expected_z, places=6)

    def test_negative_volume_in_history_raises(self) -> None:
        start = datetime(2026, 1, 26, 6, 0, tzinfo=timezone.utc)
        bars = []
        for idx in range(31):
            volume = 10.0
            if idx == 30:
                volume = 0.0
            bars.append(make_bar(start + timedelta(minutes=idx), 100.0, volume=volume))
        bars[0] = make_bar(start, 100.0, volume=-1.0)

        asof = start + timedelta(minutes=31)
        config = FeatureConfig(event_type_vocab=["CPI_US"])
        with self.assertRaises(FeatureError):
            generate_feature_vector(bars, [], asof, config, lookback_bars=len(bars))

    def test_time_features_require_utc(self) -> None:
        jst = timezone(timedelta(hours=9))
        start = datetime(2026, 1, 26, 7, 0, tzinfo=jst)
        bars = [make_bar(start + timedelta(minutes=idx), 100.0, volume=10.0) for idx in range(31)]

        asof = datetime(2026, 1, 26, 7, 31, tzinfo=timezone.utc)
        config = FeatureConfig(event_type_vocab=["CPI_US"])
        with self.assertRaises(FeatureError):
            generate_feature_vector(bars, [], asof, config, lookback_bars=len(bars))


if __name__ == "__main__":
    unittest.main()
