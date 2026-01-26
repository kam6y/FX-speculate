from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Sequence, Tuple

from .validation import MacroEvent, OhlcvBar

MAD_SCALE = 1.4826
DEFAULT_RV_WINDOW = 30
DEFAULT_EVENT_LOOKAHEAD_MINUTES = 24 * 60
DEFAULT_EVENT_DECAY_TAU_MINUTES = 60.0
DEFAULT_EVENT_TIME_TO_NEXT_MAX_MINUTES = 24 * 60

_IMPORTANCE_RANK = {"high": 3, "medium": 2, "low": 1}


class FeatureError(ValueError):
    pass


@dataclass(frozen=True)
class FeatureConfig:
    event_type_vocab: Sequence[str]
    rv_window: int = DEFAULT_RV_WINDOW
    event_lookahead_minutes: int = DEFAULT_EVENT_LOOKAHEAD_MINUTES
    event_decay_tau_minutes: float = DEFAULT_EVENT_DECAY_TAU_MINUTES
    event_time_to_next_clip_min: float = 0.0
    event_time_to_next_clip_max: float = DEFAULT_EVENT_TIME_TO_NEXT_MAX_MINUTES
    week_start: int = 0
    use_asof_for_time_features: bool = False

    def __post_init__(self) -> None:
        if not self.event_type_vocab:
            raise FeatureError("event_type_vocab must not be empty")
        if len(set(self.event_type_vocab)) != len(self.event_type_vocab):
            raise FeatureError("event_type_vocab must be unique")
        if any(not isinstance(value, str) or not value for value in self.event_type_vocab):
            raise FeatureError("event_type_vocab must contain non-empty strings")
        if list(self.event_type_vocab) != sorted(self.event_type_vocab):
            raise FeatureError("event_type_vocab must be ASCII-sorted ascending")
        if self.rv_window <= 0:
            raise FeatureError("rv_window must be > 0")
        if self.event_lookahead_minutes <= 0:
            raise FeatureError("event_lookahead_minutes must be > 0")
        if self.event_decay_tau_minutes <= 0:
            raise FeatureError("event_decay_tau_minutes must be > 0")
        if self.event_time_to_next_clip_max < self.event_time_to_next_clip_min:
            raise FeatureError("event_time_to_next_clip_max must be >= clip_min")
        if self.week_start < 0 or self.week_start > 6:
            raise FeatureError("week_start must be between 0 (Mon) and 6 (Sun)")


@dataclass(frozen=True)
class FeatureResult:
    feature_names: List[str]
    values: List[float]
    feature_map: Dict[str, float]


def build_feature_names(config: FeatureConfig) -> List[str]:
    names = [
        "return_1",
        "hl_range_norm",
        f"rv_{config.rv_window}",
        "volume_log",
        "volume_z",
        "day_sin",
        "day_cos",
        "week_sin",
        "week_cos",
    ]
    names.extend([f"event_onehot_{event_type}" for event_type in config.event_type_vocab])
    names.extend(
        [
            "event_decay_past",
            "event_time_to_next",
            "volume_missing_flag",
        ]
    )
    return names


def generate_feature_vector(
    bars: Sequence[OhlcvBar],
    macro_events: Sequence[MacroEvent],
    asof_timestamp_utc: datetime,
    config: FeatureConfig,
    lookback_bars: int,
) -> FeatureResult:
    if len(bars) < 2:
        raise FeatureError("at least two bars are required to compute return_1")
    if len(bars) < config.rv_window + 1:
        raise FeatureError(
            f"at least {config.rv_window + 1} bars are required for rv_{config.rv_window}"
        )
    if not isinstance(lookback_bars, int) or isinstance(lookback_bars, bool):
        raise FeatureError("lookback_bars must be an integer")
    if lookback_bars <= 0:
        raise FeatureError("lookback_bars must be > 0")
    if len(bars) < lookback_bars:
        raise FeatureError("bars must contain at least lookback_bars entries")

    last_bar = bars[-1]
    prev_bar = bars[-2]

    return_1 = _log_return(prev_bar.close, last_bar.close)
    hl_range_norm = _hl_range_norm(last_bar.high, last_bar.low, last_bar.close)
    rv_value = _realized_volatility(bars, config.rv_window)

    volume_bars = bars[-lookback_bars:]
    if any(bar.volume < 0 for bar in volume_bars):
        raise FeatureError("volume must be >= 0")
    volume_missing_flag = 1.0 if last_bar.volume == 0 else 0.0

    volume_log = 0.0
    volume_z = 0.0
    if volume_missing_flag == 0.0:
        volume_log = math.log(last_bar.volume + 1.0)
        volume_z = _volume_zscore(volume_bars, last_bar.volume)

    time_ref = asof_timestamp_utc if config.use_asof_for_time_features else last_bar.timestamp_utc
    day_sin, day_cos, week_sin, week_cos = _time_sin_cos(time_ref, config.week_start)

    onehot = _event_onehot(
        macro_events,
        asof_timestamp_utc,
        config.event_lookahead_minutes,
        config.event_type_vocab,
    )
    event_decay_past = _event_decay_past(
        macro_events,
        asof_timestamp_utc,
        config.event_decay_tau_minutes,
    )
    event_time_to_next = _event_time_to_next(
        macro_events,
        asof_timestamp_utc,
        config.event_time_to_next_clip_min,
        config.event_time_to_next_clip_max,
    )

    feature_map: Dict[str, float] = {
        "return_1": return_1,
        "hl_range_norm": hl_range_norm,
        f"rv_{config.rv_window}": rv_value,
        "volume_log": volume_log,
        "volume_z": volume_z,
        "day_sin": day_sin,
        "day_cos": day_cos,
        "week_sin": week_sin,
        "week_cos": week_cos,
        "event_decay_past": event_decay_past,
        "event_time_to_next": event_time_to_next,
        "volume_missing_flag": volume_missing_flag,
    }

    for event_type, value in zip(config.event_type_vocab, onehot):
        feature_map[f"event_onehot_{event_type}"] = value

    feature_names = build_feature_names(config)
    values = [feature_map[name] for name in feature_names]
    return FeatureResult(feature_names=feature_names, values=values, feature_map=feature_map)


def _log_return(prev_close: float, close: float) -> float:
    if prev_close <= 0 or close <= 0:
        raise FeatureError("close values must be > 0 for log return")
    return math.log(close / prev_close)


def _hl_range_norm(high: float, low: float, close: float) -> float:
    if close == 0:
        raise FeatureError("close must be non-zero for hl_range_norm")
    return (high - low) / close


def _realized_volatility(bars: Sequence[OhlcvBar], window: int) -> float:
    start = len(bars) - window
    if start < 1:
        raise FeatureError("not enough bars for realized volatility")
    sum_sq = 0.0
    for idx in range(start, len(bars)):
        close = bars[idx].close
        prev_close = bars[idx - 1].close
        r = _log_return(prev_close, close)
        sum_sq += r * r
    return math.sqrt(sum_sq)


def _volume_zscore(bars: Sequence[OhlcvBar], volume: float) -> float:
    volumes = [bar.volume for bar in bars]
    if not volumes:
        raise FeatureError("bars must not be empty for volume zscore")
    if any(v < 0 for v in volumes):
        raise FeatureError("volume must be >= 0")
    median = statistics.median(volumes)
    mad = statistics.median([abs(v - median) for v in volumes])
    scale = mad * MAD_SCALE
    if scale == 0:
        return 0.0
    return (volume - median) / scale


def _time_sin_cos(timestamp_utc: datetime, week_start: int) -> Tuple[float, float, float, float]:
    if timestamp_utc.tzinfo is None:
        raise FeatureError("timestamp_utc must be timezone-aware")
    offset = timestamp_utc.utcoffset()
    if offset is None:
        raise FeatureError("timestamp_utc must be timezone-aware")
    if offset != timedelta(0):
        raise FeatureError("timestamp_utc must be UTC")
    seconds_in_day = 24 * 60 * 60
    seconds_since_midnight = (
        timestamp_utc.hour * 3600 + timestamp_utc.minute * 60 + timestamp_utc.second
    )
    day_angle = 2 * math.pi * (seconds_since_midnight / seconds_in_day)
    day_sin = math.sin(day_angle)
    day_cos = math.cos(day_angle)

    weekday = timestamp_utc.weekday()
    offset_days = (weekday - week_start) % 7
    seconds_since_week_start = offset_days * seconds_in_day + seconds_since_midnight
    week_angle = 2 * math.pi * (seconds_since_week_start / (7 * seconds_in_day))
    week_sin = math.sin(week_angle)
    week_cos = math.cos(week_angle)
    return day_sin, day_cos, week_sin, week_cos


def _event_onehot(
    events: Sequence[MacroEvent],
    asof_timestamp_utc: datetime,
    lookahead_minutes: int,
    event_type_vocab: Sequence[str],
) -> List[float]:
    event_type_to_index = {event_type: idx for idx, event_type in enumerate(event_type_vocab)}
    window_end = asof_timestamp_utc + timedelta(minutes=lookahead_minutes)

    upcoming = [
        event
        for event in events
        if event.scheduled_time_utc >= asof_timestamp_utc
        and event.scheduled_time_utc <= window_end
    ]
    if not upcoming:
        return [0.0] * len(event_type_vocab)

    nearest_time = min(event.scheduled_time_utc for event in upcoming)
    same_time = [event for event in upcoming if event.scheduled_time_utc == nearest_time]

    def sort_key(event: MacroEvent) -> Tuple[int, int]:
        if event.event_type not in event_type_to_index:
            raise FeatureError(f"event_type not in vocab: {event.event_type}")
        return (-_IMPORTANCE_RANK.get(event.importance, 0), event_type_to_index[event.event_type])

    chosen = sorted(same_time, key=sort_key)[0]
    onehot = [0.0] * len(event_type_vocab)
    onehot[event_type_to_index[chosen.event_type]] = 1.0
    return onehot


def _event_decay_past(
    events: Sequence[MacroEvent],
    asof_timestamp_utc: datetime,
    tau_minutes: float,
) -> float:
    published_times = [
        event.published_at_utc
        for event in events
        if event.published_at_utc is not None and event.published_at_utc <= asof_timestamp_utc
    ]
    if not published_times:
        return 0.0
    latest = max(published_times)
    delta_minutes = (asof_timestamp_utc - latest).total_seconds() / 60.0
    return math.exp(-delta_minutes / tau_minutes)


def _event_time_to_next(
    events: Sequence[MacroEvent],
    asof_timestamp_utc: datetime,
    clip_min: float,
    clip_max: float,
) -> float:
    future_times = [
        event.scheduled_time_utc
        for event in events
        if event.scheduled_time_utc >= asof_timestamp_utc
    ]
    if not future_times:
        return clip_max
    delta_minutes = (min(future_times) - asof_timestamp_utc).total_seconds() / 60.0
    clipped = min(max(delta_minutes, clip_min), clip_max)
    return clipped
