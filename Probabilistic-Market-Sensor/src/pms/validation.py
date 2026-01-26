from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, List, Optional, Sequence, Set

HTTP_BAD_REQUEST = 400
HTTP_PAYLOAD_TOO_LARGE = 413
HTTP_UNPROCESSABLE_ENTITY = 422

SCHEMA_VERSION = "1.0"
REQUIRED_SYMBOL = "USDJPY"
REQUIRED_TIMEFRAME_SEC = 60
LOOKBACK_MIN = 128
LOOKBACK_MAX = 2048
MAX_MACRO_EVENTS = 200
HORIZONS_ALLOWED = [10, 30, 60]
PAYLOAD_LIMIT_BYTES = 2 * 1024 * 1024

ISO8601_UTC_Z_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

IMPORTANCE_ENUM = {"low", "medium", "high"}
REVISION_POLICY_ENUM = {"none", "revision_possible", "revision_expected"}
DEGRADED_REASON_ENUM = {
    "volume_missing",
    "ohlcv_gap_filled",
    "ohlcv_gap_too_long",
    "macro_events_empty",
}


class ValidationError(ValueError):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


class ValidationConfigError(RuntimeError):
    pass


@dataclass
class ValidationConfig:
    event_type_allowlist: Optional[Set[str]] = None
    enforce_event_type_enum: bool = True
    payload_limit_bytes: int = PAYLOAD_LIMIT_BYTES

    def __post_init__(self) -> None:
        if self.enforce_event_type_enum and not self.event_type_allowlist:
            raise ValidationConfigError(
                "event_type_allowlist is required when enforce_event_type_enum is True"
            )


@dataclass
class RiskFlags:
    data_integrity_warning: bool = False
    degraded: bool = False
    degraded_reasons: List[str] = field(default_factory=list)

    def add_reason(self, reason: str) -> None:
        if reason not in DEGRADED_REASON_ENUM:
            raise ValueError(f"unknown degraded reason: {reason}")
        if reason not in self.degraded_reasons:
            self.degraded_reasons.append(reason)
        self.data_integrity_warning = True
        self.degraded = True


@dataclass(frozen=True)
class OhlcvBar:
    timestamp_utc: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class MacroEvent:
    event_type: str
    scheduled_time_utc: datetime
    importance: str
    revision_policy: str
    published_at_utc: Optional[datetime] = None
    actual: Optional[float] = None
    forecast: Optional[float] = None
    previous: Optional[float] = None
    unit: Optional[str] = None


@dataclass(frozen=True)
class ValidationResult:
    schema_version: str
    symbol: str
    timeframe_sec: int
    asof_timestamp_utc: datetime
    lookback_bars: int
    ohlcv: List[OhlcvBar]
    macro_events: List[MacroEvent]
    horizons_min: Optional[List[int]]
    inference_id: Optional[str]
    debug: bool
    risk_flags: RiskFlags
    payload_size_bytes: int


def validate_infer_request(payload: Any, config: ValidationConfig) -> ValidationResult:
    if not isinstance(payload, dict):
        raise ValidationError(HTTP_BAD_REQUEST, "payload must be a JSON object")

    payload_size_bytes = _estimate_payload_size_bytes(payload)
    if payload_size_bytes > config.payload_limit_bytes:
        raise ValidationError(
            HTTP_PAYLOAD_TOO_LARGE,
            f"payload exceeds {config.payload_limit_bytes} bytes after decoding",
        )

    schema_version = _require_str(
        _require_field(payload, "schema_version", HTTP_BAD_REQUEST),
        "schema_version",
        HTTP_BAD_REQUEST,
    )
    if schema_version != SCHEMA_VERSION:
        raise ValidationError(
            HTTP_UNPROCESSABLE_ENTITY,
            f"schema_version must be {SCHEMA_VERSION}",
        )

    symbol = _require_str(
        _require_field(payload, "symbol", HTTP_BAD_REQUEST),
        "symbol",
        HTTP_BAD_REQUEST,
    )
    if symbol != REQUIRED_SYMBOL:
        raise ValidationError(
            HTTP_UNPROCESSABLE_ENTITY,
            f"symbol must be {REQUIRED_SYMBOL}",
        )

    timeframe_sec = _require_int(
        _require_field(payload, "timeframe_sec", HTTP_BAD_REQUEST),
        "timeframe_sec",
        HTTP_BAD_REQUEST,
    )
    if timeframe_sec != REQUIRED_TIMEFRAME_SEC:
        raise ValidationError(
            HTTP_UNPROCESSABLE_ENTITY,
            f"timeframe_sec must be {REQUIRED_TIMEFRAME_SEC}",
        )

    asof_timestamp_raw = _require_field(payload, "asof_timestamp_utc", HTTP_BAD_REQUEST)
    asof_timestamp_utc = _parse_utc_timestamp(
        asof_timestamp_raw,
        "asof_timestamp_utc",
        HTTP_BAD_REQUEST,
    )

    lookback_bars = _require_int(
        _require_field(payload, "lookback_bars", HTTP_BAD_REQUEST),
        "lookback_bars",
        HTTP_BAD_REQUEST,
    )
    if lookback_bars < LOOKBACK_MIN or lookback_bars > LOOKBACK_MAX:
        raise ValidationError(
            HTTP_UNPROCESSABLE_ENTITY,
            f"lookback_bars must be between {LOOKBACK_MIN} and {LOOKBACK_MAX}",
        )

    ohlcv_raw = _require_field(payload, "ohlcv", HTTP_BAD_REQUEST)
    if not isinstance(ohlcv_raw, list):
        raise ValidationError(HTTP_BAD_REQUEST, "ohlcv must be an array")

    macro_events_raw = _require_field(payload, "macro_events", HTTP_BAD_REQUEST)
    if not isinstance(macro_events_raw, list):
        raise ValidationError(HTTP_BAD_REQUEST, "macro_events must be an array")

    horizons_min = payload.get("horizons_min")
    if horizons_min is not None:
        if not isinstance(horizons_min, list):
            raise ValidationError(HTTP_BAD_REQUEST, "horizons_min must be an array")
        horizons_min = _validate_horizons(horizons_min)

    inference_id = payload.get("inference_id")
    if inference_id is not None and not isinstance(inference_id, str):
        raise ValidationError(HTTP_BAD_REQUEST, "inference_id must be a string")

    debug = payload.get("debug", False)
    if not isinstance(debug, bool):
        raise ValidationError(HTTP_BAD_REQUEST, "debug must be a boolean")

    ohlcv = _validate_ohlcv(
        ohlcv_raw,
        lookback_bars,
        asof_timestamp_utc,
        timeframe_sec,
    )

    if len(macro_events_raw) > MAX_MACRO_EVENTS:
        raise ValidationError(
            HTTP_UNPROCESSABLE_ENTITY,
            f"macro_events must have <= {MAX_MACRO_EVENTS} items",
        )

    risk_flags = RiskFlags()
    macro_events = _validate_macro_events(
        macro_events_raw,
        asof_timestamp_utc,
        config,
        risk_flags,
    )

    return ValidationResult(
        schema_version=schema_version,
        symbol=symbol,
        timeframe_sec=timeframe_sec,
        asof_timestamp_utc=asof_timestamp_utc,
        lookback_bars=lookback_bars,
        ohlcv=ohlcv,
        macro_events=macro_events,
        horizons_min=horizons_min,
        inference_id=inference_id,
        debug=debug,
        risk_flags=risk_flags,
        payload_size_bytes=payload_size_bytes,
    )


def _estimate_payload_size_bytes(payload: Any) -> int:
    try:
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            HTTP_BAD_REQUEST,
            f"payload must be JSON-serializable: {exc}",
        ) from exc
    return len(encoded)


def _validate_horizons(values: Sequence[Any]) -> List[int]:
    horizons: List[int] = []
    for idx, value in enumerate(values):
        if not _is_int(value):
            raise ValidationError(
                HTTP_UNPROCESSABLE_ENTITY,
                f"horizons_min[{idx}] must be an integer",
            )
        horizons.append(int(value))
    if horizons != HORIZONS_ALLOWED:
        raise ValidationError(
            HTTP_UNPROCESSABLE_ENTITY,
            f"horizons_min must be {HORIZONS_ALLOWED}",
        )
    return horizons


def _validate_ohlcv(
    records: Sequence[Any],
    lookback_bars: int,
    asof_timestamp_utc: datetime,
    timeframe_sec: int,
) -> List[OhlcvBar]:
    if len(records) != lookback_bars:
        raise ValidationError(
            HTTP_UNPROCESSABLE_ENTITY,
            "ohlcv length must equal lookback_bars",
        )

    bars: List[OhlcvBar] = []
    previous_ts: Optional[datetime] = None

    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValidationError(
                HTTP_UNPROCESSABLE_ENTITY,
                f"ohlcv[{idx}] must be an object",
            )

        timestamp_raw = _require_field(record, "timestamp_utc", HTTP_UNPROCESSABLE_ENTITY)
        timestamp_utc = _parse_utc_timestamp(
            timestamp_raw,
            f"ohlcv[{idx}].timestamp_utc",
            HTTP_UNPROCESSABLE_ENTITY,
        )

        open_value = _require_number(
            _require_field(record, "open", HTTP_UNPROCESSABLE_ENTITY),
            f"ohlcv[{idx}].open",
        )
        high_value = _require_number(
            _require_field(record, "high", HTTP_UNPROCESSABLE_ENTITY),
            f"ohlcv[{idx}].high",
        )
        low_value = _require_number(
            _require_field(record, "low", HTTP_UNPROCESSABLE_ENTITY),
            f"ohlcv[{idx}].low",
        )
        close_value = _require_number(
            _require_field(record, "close", HTTP_UNPROCESSABLE_ENTITY),
            f"ohlcv[{idx}].close",
        )
        volume_value = _require_number(
            _require_field(record, "volume", HTTP_UNPROCESSABLE_ENTITY),
            f"ohlcv[{idx}].volume",
        )

        if previous_ts is not None and timestamp_utc <= previous_ts:
            raise ValidationError(
                HTTP_UNPROCESSABLE_ENTITY,
                "ohlcv timestamps must be strictly increasing",
            )
        previous_ts = timestamp_utc

        if high_value < max(open_value, close_value):
            raise ValidationError(
                HTTP_UNPROCESSABLE_ENTITY,
                f"ohlcv[{idx}] high must be >= max(open, close)",
            )
        if low_value > min(open_value, close_value):
            raise ValidationError(
                HTTP_UNPROCESSABLE_ENTITY,
                f"ohlcv[{idx}] low must be <= min(open, close)",
            )
        if high_value < low_value:
            raise ValidationError(
                HTTP_UNPROCESSABLE_ENTITY,
                f"ohlcv[{idx}] high must be >= low",
            )

        bars.append(
            OhlcvBar(
                timestamp_utc=timestamp_utc,
                open=open_value,
                high=high_value,
                low=low_value,
                close=close_value,
                volume=volume_value,
            )
        )

    expected_last = asof_timestamp_utc - timedelta(seconds=timeframe_sec)
    if not bars or bars[-1].timestamp_utc != expected_last:
        raise ValidationError(
            HTTP_UNPROCESSABLE_ENTITY,
            "last ohlcv timestamp must equal asof_timestamp_utc - timeframe_sec",
        )

    return bars


def _validate_macro_events(
    records: Sequence[Any],
    asof_timestamp_utc: datetime,
    config: ValidationConfig,
    risk_flags: RiskFlags,
) -> List[MacroEvent]:
    if not records:
        risk_flags.add_reason("macro_events_empty")
        return []

    if config.enforce_event_type_enum and not config.event_type_allowlist:
        raise ValidationConfigError(
            "event_type_allowlist is required for macro event validation"
        )

    events: List[MacroEvent] = []
    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValidationError(
                HTTP_UNPROCESSABLE_ENTITY,
                f"macro_events[{idx}] must be an object",
            )

        event_type = _require_str(
            _require_field(record, "event_type", HTTP_UNPROCESSABLE_ENTITY),
            f"macro_events[{idx}].event_type",
            HTTP_UNPROCESSABLE_ENTITY,
        )
        if config.enforce_event_type_enum and event_type not in config.event_type_allowlist:
            raise ValidationError(
                HTTP_UNPROCESSABLE_ENTITY,
                f"macro_events[{idx}].event_type is not allowed: {event_type}",
            )

        scheduled_raw = _require_field(
            record,
            "scheduled_time_utc",
            HTTP_UNPROCESSABLE_ENTITY,
        )
        scheduled_time_utc = _parse_utc_timestamp(
            scheduled_raw,
            f"macro_events[{idx}].scheduled_time_utc",
            HTTP_UNPROCESSABLE_ENTITY,
        )

        importance = _require_str(
            _require_field(record, "importance", HTTP_UNPROCESSABLE_ENTITY),
            f"macro_events[{idx}].importance",
            HTTP_UNPROCESSABLE_ENTITY,
        ).lower()
        if importance not in IMPORTANCE_ENUM:
            raise ValidationError(
                HTTP_UNPROCESSABLE_ENTITY,
                f"macro_events[{idx}].importance must be one of {sorted(IMPORTANCE_ENUM)}",
            )

        revision_policy = _require_str(
            _require_field(record, "revision_policy", HTTP_UNPROCESSABLE_ENTITY),
            f"macro_events[{idx}].revision_policy",
            HTTP_UNPROCESSABLE_ENTITY,
        ).lower()
        if revision_policy not in REVISION_POLICY_ENUM:
            raise ValidationError(
                HTTP_UNPROCESSABLE_ENTITY,
                f"macro_events[{idx}].revision_policy must be one of {sorted(REVISION_POLICY_ENUM)}",
            )

        published_at_utc = None
        if "published_at_utc" in record:
            published_raw = record.get("published_at_utc")
            if published_raw is None:
                raise ValidationError(
                    HTTP_UNPROCESSABLE_ENTITY,
                    f"macro_events[{idx}].published_at_utc must be a string when provided",
                )
            published_at_utc = _parse_utc_timestamp(
                published_raw,
                f"macro_events[{idx}].published_at_utc",
                HTTP_UNPROCESSABLE_ENTITY,
            )
            if published_at_utc > asof_timestamp_utc:
                raise ValidationError(
                    HTTP_UNPROCESSABLE_ENTITY,
                    f"macro_events[{idx}].published_at_utc must be <= asof_timestamp_utc",
                )

        actual = _optional_number(record, "actual", idx)
        forecast = _optional_number(record, "forecast", idx)
        previous = _optional_number(record, "previous", idx)

        unit = None
        if "unit" in record:
            unit_raw = record.get("unit")
            if unit_raw is None or not isinstance(unit_raw, str):
                raise ValidationError(
                    HTTP_UNPROCESSABLE_ENTITY,
                    f"macro_events[{idx}].unit must be a string when provided",
                )
            unit = unit_raw

        if actual is not None:
            if published_at_utc is None:
                raise ValidationError(
                    HTTP_UNPROCESSABLE_ENTITY,
                    f"macro_events[{idx}].actual requires published_at_utc",
                )
            if scheduled_time_utc > asof_timestamp_utc:
                raise ValidationError(
                    HTTP_UNPROCESSABLE_ENTITY,
                    f"macro_events[{idx}].actual is not allowed for future scheduled_time_utc",
                )

        events.append(
            MacroEvent(
                event_type=event_type,
                scheduled_time_utc=scheduled_time_utc,
                importance=importance,
                revision_policy=revision_policy,
                published_at_utc=published_at_utc,
                actual=actual,
                forecast=forecast,
                previous=previous,
                unit=unit,
            )
        )

    return events


def _optional_number(record: dict, key: str, idx: int) -> Optional[float]:
    if key not in record:
        return None
    value = record.get(key)
    if value is None:
        raise ValidationError(
            HTTP_UNPROCESSABLE_ENTITY,
            f"macro_events[{idx}].{key} must be a number when provided",
        )
    return _require_number(value, f"macro_events[{idx}].{key}")


def _require_field(container: dict, key: str, status_code: int) -> Any:
    if key not in container:
        raise ValidationError(status_code, f"missing field: {key}")
    return container[key]


def _require_str(value: Any, field_name: str, status_code: int) -> str:
    if not isinstance(value, str):
        raise ValidationError(status_code, f"{field_name} must be a string")
    return value


def _require_int(value: Any, field_name: str, status_code: int) -> int:
    if not _is_int(value):
        raise ValidationError(status_code, f"{field_name} must be an integer")
    return int(value)


def _require_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(
            HTTP_UNPROCESSABLE_ENTITY,
            f"{field_name} must be a number",
        )
    number = float(value)
    if not math.isfinite(number):
        raise ValidationError(
            HTTP_UNPROCESSABLE_ENTITY,
            f"{field_name} must be finite",
        )
    return number


def _parse_utc_timestamp(value: Any, field_name: str, type_status: int) -> datetime:
    if not isinstance(value, str):
        raise ValidationError(type_status, f"{field_name} must be a string")
    try:
        parsed = datetime.strptime(value, ISO8601_UTC_Z_FORMAT)
    except ValueError as exc:
        raise ValidationError(
            HTTP_UNPROCESSABLE_ENTITY,
            f"{field_name} must be ISO8601 UTC with Z and seconds=0",
        ) from exc
    return parsed.replace(tzinfo=timezone.utc)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)
