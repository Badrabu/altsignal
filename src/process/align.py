"""Alignment of every signal to the project's analysis grain: MONTHLY.

Three situations are handled:

1. Native monthly series (MSRS, VIIRS, FRED): nothing to do except check the
   rows are aligned to calendar months (``clean.py`` does that). Their ``grain``
   stays "monthly".

2. Weekly Indeed postings: counted again per CALENDAR MONTH straight from the
   deduplicated postings (each posting is dated by the day we first saw it),
   instead of adding up weeks that straddle two months. The monthly rows keep
   the same ``signal_id`` as the weekly rows and use ``grain = "monthly"``; the
   weekly rows stay in the table with ``grain = "weekly"``. The quality score is
   the share of the month's days on which Indeed was actually scraped.

3. Series published less often than monthly (e.g. quarterly): the value is
   REPEATED across the months it covers and every repeated row is tagged with
   ``grain = "quarterly_repeated"`` so nobody mistakes it for a real monthly
   observation. Nothing is carried beyond the period the value covers, so this
   is not imputation. (No current source is quarterly; the function is ready
   for later ones.)
"""

from __future__ import annotations

import calendar
from datetime import date

import pandas as pd

import config
from src.ingest.indeed import sector_slug
from src.normalize import add_normalized_values


def month_start(day: date) -> pd.Timestamp:
    """Return the first day of the month containing ``day``.

    Args:
        day: Any date.

    Returns:
        A ``Timestamp`` for the 1st of that month.
    """
    return pd.Timestamp(day.year, day.month, 1)


def indeed_monthly_rows(
    postings: pd.DataFrame,
    state_name: str,
    metros: list[config.Metro],
    observed_days: set[date],
) -> pd.DataFrame:
    """Count deduplicated Indeed postings per calendar month, metro and sector.

    Only months in which Indeed was scraped at least once get rows. In such a
    month a metro/sector with no new postings is a real zero (we looked and saw
    none). ``quality_score`` is observed days divided by days in the month, so
    a month scraped on 9 of 30 days scores 0.3.

    Series produced (``signal_id``, all with unit "postings"):
        ``indeed_postings_<sector>``  per metro and sector
        ``indeed_postings_total``     per metro, and for the whole state

    Args:
        postings: Deduplicated postings with ``first_seen`` (YYYY-MM-DD),
            ``metro`` (the metro ``name``) and ``sector`` columns.
        state_name: Full state name (geo_name of the state-level total).
        metros: The state's metros.
        observed_days: Dates on which a scrape ran.

    Returns:
        Signal rows (without ``normalized_value``); empty if never scraped.
    """
    if not observed_days:
        return pd.DataFrame()

    days_seen: dict[pd.Timestamp, int] = {}
    for day in observed_days:
        key = month_start(day)
        days_seen[key] = days_seen.get(key, 0) + 1

    posts = postings.copy()
    posts["month"] = [month_start(date.fromisoformat(d)) for d in posts["first_seen"]]
    counts = posts.groupby(["month", "metro", "sector"]).size().to_dict()
    pairs = sorted({(metro, sector) for (_, metro, sector) in counts})

    records = []
    for month in sorted(days_seen):
        days_in_month = calendar.monthrange(month.year, month.month)[1]
        base = {
            "source": config.SOURCE_INDEED,
            "period_start": month,
            "period_end": month + pd.offsets.MonthEnd(0),
            "unit": "postings",
            "quality_score": min(days_seen[month] / days_in_month, 1.0),
            "grain": "monthly",
        }
        state_total = 0
        for metro_obj in metros:
            metro_total = 0
            for metro_name, sector in pairs:
                if metro_name != metro_obj.name:
                    continue
                value = counts.get((month, metro_name, sector), 0)
                metro_total += value
                records.append({**base, "signal_id": f"indeed_postings_{sector_slug(sector)}",
                                "geo_level": "metro", "geo_name": metro_obj.geo_name, "value": value})
            state_total += metro_total
            records.append({**base, "signal_id": "indeed_postings_total",
                            "geo_level": "metro", "geo_name": metro_obj.geo_name, "value": metro_total})
        records.append({**base, "signal_id": "indeed_postings_total",
                        "geo_level": "state", "geo_name": state_name, "value": state_total})
    return pd.DataFrame(records)


def repeat_to_monthly(quarterly: pd.DataFrame) -> pd.DataFrame:
    """Spread quarterly rows across their three months, tagged as repeated.

    The z-score is computed on the native QUARTERLY series first (baseline of
    the previous 4 quarters, about 12 months) and then repeated with the value,
    so repeated rows never create fake extra history.

    Args:
        quarterly: Signal rows with ``grain = "quarterly"`` and ``period_start``
            on the first day of a quarter.

    Returns:
        Three monthly rows per quarterly row with ``grain = "quarterly_repeated"``
        (empty if the input is empty).
    """
    if quarterly.empty:
        return quarterly.copy()
    scored = add_normalized_values(quarterly)
    pieces = []
    for offset in range(3):
        piece = scored.copy()
        piece["period_start"] = scored["period_start"] + pd.DateOffset(months=offset)
        piece["period_end"] = piece["period_start"] + pd.offsets.MonthEnd(0)
        piece["grain"] = "quarterly_repeated"
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)
