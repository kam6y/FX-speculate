from __future__ import annotations

from datetime import datetime, timedelta, time
from typing import List, Sequence

from .validation import (
    HTTP_UNPROCESSABLE_ENTITY,
    OhlcvBar,
    RiskFlags,
    ValidationError,
)

MAX_CONTIGUOUS_GAP_BARS = 60
SHORT_GAP_MAX_BARS = 3

_WEEKLY_CLOSE_START_UTC = time(21, 0, 0)
_WEEKLY_CLOSE_END_UTC = time(21, 59, 59)


def is_market_closed(timestamp_utc: datetime) -> bool:
    """Return True if the timestamp is inside the weekly close window (UTC)."""
    weekday = timestamp_utc.weekday()  # Monday=0 ... Sunday=6
    if weekday == 4:  # Friday
        return timestamp_utc.time() >= _WEEKLY_CLOSE_START_UTC
    if weekday == 5:  # Saturday
        return True
    if weekday == 6:  # Sunday
        return timestamp_utc.time() <= _WEEKLY_CLOSE_END_UTC
    return False


def fill_ohlcv_gaps(
    bars: Sequence[OhlcvBar],
    lookback_bars: int,
    timeframe_sec: int,
    risk_flags: RiskFlags,
) -> List[OhlcvBar]:
    """Fill timestamp gaps after validation, update risk flags, and trim to lookback."""
    if not bars:
        return []

    if any(bar.volume == 0 for bar in bars):
        risk_flags.add_reason("volume_missing")

    filled: List[OhlcvBar] = [bars[0]]
    has_short_gap = False
    has_long_gap = False
    gap_len = 0

    def finalize_gap(length: int) -> None:
        nonlocal has_short_gap, has_long_gap
        if length <= 0:
            return
        if length > MAX_CONTIGUOUS_GAP_BARS:
            raise ValidationError(
                HTTP_UNPROCESSABLE_ENTITY,
                "ohlcv gap exceeds 60 bars",
            )
        if length <= SHORT_GAP_MAX_BARS:
            has_short_gap = True
        else:
            has_long_gap = True

    step = timedelta(seconds=timeframe_sec)

    for current in bars[1:]:
        prev = filled[-1]
        expected_ts = prev.timestamp_utc + step

        while expected_ts < current.timestamp_utc:
            if is_market_closed(expected_ts):
                if gap_len > 0:
                    finalize_gap(gap_len)
                    gap_len = 0
                expected_ts += step
                continue

            gap_len += 1
            if gap_len > MAX_CONTIGUOUS_GAP_BARS:
                raise ValidationError(
                    HTTP_UNPROCESSABLE_ENTITY,
                    "ohlcv gap exceeds 60 bars",
                )

            close_value = filled[-1].close
            filled.append(
                OhlcvBar(
                    timestamp_utc=expected_ts,
                    open=close_value,
                    high=close_value,
                    low=close_value,
                    close=close_value,
                    volume=0.0,
                )
            )
            expected_ts += step

        finalize_gap(gap_len)
        gap_len = 0
        filled.append(current)

    if has_short_gap:
        risk_flags.add_reason("ohlcv_gap_filled")
    if has_long_gap:
        risk_flags.add_reason("ohlcv_gap_too_long")

    if len(filled) > lookback_bars:
        filled = filled[-lookback_bars:]

    return filled
