import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from pms.validation import (  # noqa: E402
    ValidationConfig,
    ValidationError,
    validate_infer_request,
)

ISO_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def isoformat_z(dt: datetime) -> str:
    return dt.strftime(ISO_FORMAT)


def parse_iso(value: str) -> datetime:
    return datetime.strptime(value, ISO_FORMAT).replace(tzinfo=timezone.utc)


def build_payload(
    lookback_bars: int = 128,
    macro_events=None,
    horizons_min=None,
) -> dict:
    start = datetime(2026, 1, 26, 0, 0, tzinfo=timezone.utc)
    ohlcv = []
    for idx in range(lookback_bars):
        timestamp = start + timedelta(minutes=idx)
        open_value = 150.0 + idx * 0.01
        close_value = open_value + 0.005
        high_value = close_value + 0.01
        low_value = open_value - 0.01
        ohlcv.append(
            {
                "timestamp_utc": isoformat_z(timestamp),
                "open": open_value,
                "high": high_value,
                "low": low_value,
                "close": close_value,
                "volume": 1000.0,
            }
        )

    asof_timestamp = start + timedelta(minutes=lookback_bars)
    payload = {
        "schema_version": "1.0",
        "symbol": "USDJPY",
        "timeframe_sec": 60,
        "asof_timestamp_utc": isoformat_z(asof_timestamp),
        "lookback_bars": lookback_bars,
        "ohlcv": ohlcv,
        "macro_events": macro_events if macro_events is not None else [],
    }
    if horizons_min is not None:
        payload["horizons_min"] = horizons_min
    return payload


def build_macro_event(scheduled: datetime) -> dict:
    return {
        "event_type": "CPI_US",
        "scheduled_time_utc": isoformat_z(scheduled),
        "importance": "high",
        "revision_policy": "none",
        "published_at_utc": isoformat_z(scheduled),
        "actual": 3.1,
        "forecast": 3.0,
        "previous": 2.9,
        "unit": "percent",
    }


class ValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ValidationConfig(event_type_allowlist={"CPI_US"})

    def test_valid_payload_with_empty_macro_events_sets_risk_flags(self) -> None:
        payload = build_payload()
        result = validate_infer_request(payload, self.config)
        self.assertTrue(result.risk_flags.data_integrity_warning)
        self.assertTrue(result.risk_flags.degraded)
        self.assertIn("macro_events_empty", result.risk_flags.degraded_reasons)

    def test_valid_payload_with_macro_event(self) -> None:
        payload = build_payload()
        asof = parse_iso(payload["asof_timestamp_utc"])
        scheduled = asof - timedelta(minutes=10)
        payload["macro_events"] = [build_macro_event(scheduled)]
        result = validate_infer_request(payload, self.config)
        self.assertFalse(result.risk_flags.degraded)
        self.assertEqual(result.macro_events[0].event_type, "CPI_US")

    def test_macro_events_missing_is_400(self) -> None:
        payload = build_payload()
        del payload["macro_events"]
        with self.assertRaises(ValidationError) as ctx:
            validate_infer_request(payload, self.config)
        self.assertEqual(ctx.exception.status_code, 400)

    def test_horizons_min_invalid_is_422(self) -> None:
        payload = build_payload(horizons_min=[10, 60, 30])
        with self.assertRaises(ValidationError) as ctx:
            validate_infer_request(payload, self.config)
        self.assertEqual(ctx.exception.status_code, 422)

    def test_ohlcv_last_timestamp_mismatch_is_422(self) -> None:
        payload = build_payload()
        asof = parse_iso(payload["asof_timestamp_utc"])
        payload["asof_timestamp_utc"] = isoformat_z(asof + timedelta(minutes=1))
        with self.assertRaises(ValidationError) as ctx:
            validate_infer_request(payload, self.config)
        self.assertEqual(ctx.exception.status_code, 422)

    def test_payload_too_large_is_413(self) -> None:
        payload = build_payload()
        small_config = ValidationConfig(
            event_type_allowlist={"CPI_US"},
            payload_limit_bytes=10,
        )
        with self.assertRaises(ValidationError) as ctx:
            validate_infer_request(payload, small_config)
        self.assertEqual(ctx.exception.status_code, 413)


if __name__ == "__main__":
    unittest.main()
