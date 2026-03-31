"""経済イベントカレンダー。

FOMC、日銀会合、米雇用統計(NFP)、CPI、GDP、ISM、小売売上、
ジャクソンホール会議の日程を管理し、イベント特徴量を算出する。
"""

import pandas as pd
import numpy as np

# --- 不規則イベント: 年次リストで手動管理 ---
FOMC_DATES = {
    2016: ["01-27", "03-16", "04-27", "06-15", "07-27", "09-21", "11-02", "12-14"],
    2017: ["02-01", "03-15", "05-03", "06-14", "07-26", "09-20", "11-01", "12-13"],
    2018: ["01-31", "03-21", "05-02", "06-13", "08-01", "09-26", "11-08", "12-19"],
    2019: ["01-30", "03-20", "05-01", "06-19", "07-31", "09-18", "10-30", "12-11"],
    2020: ["01-29", "03-03", "03-15", "04-29", "06-10", "07-29", "09-16", "11-05", "12-16"],
    2021: ["01-27", "03-17", "04-28", "06-16", "07-28", "09-22", "11-03", "12-15"],
    2022: ["01-26", "03-16", "05-04", "06-15", "07-27", "09-21", "11-02", "12-14"],
    2023: ["02-01", "03-22", "05-03", "06-14", "07-26", "09-20", "11-01", "12-13"],
    2024: ["01-31", "03-20", "05-01", "06-12", "07-31", "09-18", "11-07", "12-18"],
    2025: ["01-29", "03-19", "05-07", "06-18", "07-30", "09-17", "10-29", "12-17"],
    2026: ["01-28", "03-18", "04-29", "06-17", "07-29", "09-16", "11-04", "12-16"],
}

BOJ_DATES = {
    2016: ["01-29", "03-15", "04-28", "06-16", "07-29", "09-21", "11-01", "12-20"],
    2017: ["01-31", "03-16", "04-27", "06-16", "07-20", "09-21", "10-31", "12-21"],
    2018: ["01-23", "03-09", "04-27", "06-15", "07-31", "09-19", "10-31", "12-20"],
    2019: ["01-23", "03-15", "04-25", "06-20", "07-30", "09-19", "10-31", "12-19"],
    2020: ["01-21", "03-16", "04-27", "06-16", "07-15", "09-17", "10-29", "12-18"],
    2021: ["01-21", "03-19", "04-27", "06-18", "07-16", "09-22", "10-28", "12-17"],
    2022: ["01-18", "03-18", "04-28", "06-17", "07-21", "09-22", "10-28", "12-20"],
    2023: ["01-18", "03-10", "04-28", "06-16", "07-28", "09-22", "10-31", "12-19"],
    2024: ["01-23", "03-19", "04-26", "06-14", "07-31", "09-20", "10-31", "12-19"],
    2025: ["01-24", "03-14", "05-01", "06-17", "07-31", "09-19", "10-30", "12-19"],
    2026: ["01-22", "03-13", "04-28", "06-16", "07-16", "09-17", "10-29", "12-18"],
}

JACKSON_HOLE = {
    2016: ["08-26"], 2017: ["08-25"], 2018: ["08-24"], 2019: ["08-23"],
    2020: ["08-27"], 2021: ["08-27"], 2022: ["08-26"], 2023: ["08-25"],
    2024: ["08-23"], 2025: ["08-22"], 2026: ["08-28"],
}


def _first_friday(year: int, month: int) -> pd.Timestamp:
    """指定年月の第1金曜日を返す。"""
    first = pd.Timestamp(year=year, month=month, day=1)
    offset = (4 - first.dayofweek) % 7
    return first + pd.Timedelta(days=offset)


def _first_business_day(year: int, month: int) -> pd.Timestamp:
    """指定年月の第1営業日を返す。"""
    first = pd.Timestamp(year=year, month=month, day=1)
    if first.dayofweek >= 5:
        first += pd.offsets.BDay(1)
    return first


def _mid_month(year: int, month: int, day: int = 13) -> pd.Timestamp:
    """指定年月の中旬営業日を返す。"""
    d = pd.Timestamp(year=year, month=month, day=day)
    if d.dayofweek >= 5:
        d += pd.offsets.BDay(1)
    return d


def get_all_event_dates(year: int) -> pd.DataFrame:
    """指定年の全経済イベント日程を返す。

    Returns:
        DataFrame with columns: date (Timestamp), event_type (str)
    """
    events = []

    # FOMC
    for md in FOMC_DATES.get(year, []):
        events.append((pd.Timestamp(f"{year}-{md}"), "FOMC"))

    # 日銀
    for md in BOJ_DATES.get(year, []):
        events.append((pd.Timestamp(f"{year}-{md}"), "BOJ"))

    # ジャクソンホール
    for md in JACKSON_HOLE.get(year, []):
        events.append((pd.Timestamp(f"{year}-{md}"), "JACKSON_HOLE"))

    # NFP: 毎月第1金曜日
    for m in range(1, 13):
        events.append((_first_friday(year, m), "NFP"))

    # CPI: 毎月中旬
    for m in range(1, 13):
        events.append((_mid_month(year, m, 13), "CPI"))

    # GDP: 四半期末月の月末付近
    for m in [1, 4, 7, 10]:
        d = pd.Timestamp(year=year, month=m, day=28)
        if d.dayofweek >= 5:
            d -= pd.offsets.BDay(1)
        events.append((d, "GDP"))

    # ISM: 毎月第1営業日
    for m in range(1, 13):
        events.append((_first_business_day(year, m), "ISM"))

    # 小売売上: 毎月中旬
    for m in range(1, 13):
        events.append((_mid_month(year, m, 15), "RETAIL"))

    df = pd.DataFrame(events, columns=["date", "event_type"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _get_events_for_range(start_year: int, end_year: int) -> pd.DataFrame:
    """複数年のイベントを結合して返す。"""
    frames = [get_all_event_dates(y) for y in range(start_year, end_year + 1)]
    return pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)


def compute_event_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """日付インデックスに対してイベント特徴量を算出する。

    Args:
        index: 営業日の DatetimeIndex

    Returns:
        DataFrame with columns:
            days_to_next_major_event, days_from_last_major_event,
            event_type_next, is_event_day, event_density_past_5d
    """
    start_year = index.min().year - 1
    end_year = index.max().year + 1
    all_events = _get_events_for_range(start_year, end_year)
    event_dates = all_events["date"].values

    results = []
    for d in index:
        d_ts = pd.Timestamp(d)

        # days_to_next_major_event
        future = all_events[all_events["date"] >= d_ts]
        if len(future) > 0:
            next_event_date = future.iloc[0]["date"]
            next_event_type = future.iloc[0]["event_type"]
            days_to = np.busday_count(
                d_ts.date(), pd.Timestamp(next_event_date).date()
            )
        else:
            days_to = 30
            next_event_type = "NONE"

        # days_from_last_major_event
        past = all_events[all_events["date"] <= d_ts]
        if len(past) > 0:
            last_event_date = past.iloc[-1]["date"]
            days_from = np.busday_count(
                pd.Timestamp(last_event_date).date(), d_ts.date()
            )
        else:
            days_from = 30

        # is_event_day
        is_event = int(d_ts.normalize() in set(pd.DatetimeIndex(event_dates).normalize()))

        # event_density_past_5d: t-5d 〜 t-1d のイベント数
        past_5d_start = d_ts - pd.offsets.BDay(5)
        past_5d_events = all_events[
            (all_events["date"] >= past_5d_start) & (all_events["date"] < d_ts)
        ]
        density = len(past_5d_events)

        results.append({
            "days_to_next_major_event": max(days_to, 0),
            "days_from_last_major_event": max(days_from, 0),
            "event_type_next": next_event_type,
            "is_event_day": is_event,
            "event_density_past_5d": density,
        })

    return pd.DataFrame(results, index=index)
