"""Where do the signals disagree? Divergence rules and the plain-language summary.

The comparison uses z-scores only (each signal against its own previous 12
months), because raw values of different units must never be compared. Both the
detection and the wording are fixed rules over the numbers (no LLM), so the
same data always give the same text.

Rules, applied to a pair of signals A and B at the latest month where both
have a z-score:

* **Level divergence**: the z-scores have opposite signs and both are at least
  ``config.DIVERGENCE_MIN_Z`` (0.5) standard deviations from zero.
* **Trend divergence**: the 3-month change of each z-score has opposite signs
  and both changes are at least ``config.DIVERGENCE_MIN_Z`` in size.
* **Severity**: the gap ``z_A - z_B`` is "strong" at ``config.DIVERGENCE_STRONG_GAP``
  (2.0) or more, otherwise "moderate".
* A pair is **neutral** when it does not diverge and at least one signal is within
  the noise band, and **agrees** when both are clearly on the same side.
* A pair with no month in common has **no overlap** and is not judged.
"""

from __future__ import annotations

import pandas as pd

import config

NO_OVERLAP, AGREE, NEUTRAL, DIVERGE = "no overlap", "agree", "neutral", "diverge"
TREND_LOOKBACK_MONTHS = 3

# One headline series per source for the across-source comparison.
HEADLINE_SIGNALS = ["msrs_yoy_TOTAL", "indeed_postings_total", "viirs_avg_radiance", "card_spending_index"]
REFERENCE_SIGNAL = "msrs_yoy_TOTAL"


def compare_pair(z_a: pd.Series, z_b: pd.Series, min_z: float | None = None,
                 strong_gap: float | None = None) -> dict:
    """Apply the divergence rules to two z-score series.

    Args:
        z_a: Z-scores of signal A indexed by month start (``NaN`` = missing).
        z_b: Z-scores of signal B, same index type.
        min_z: Noise band. Defaults to ``config.DIVERGENCE_MIN_Z``.
        strong_gap: Gap for "strong". Defaults to ``config.DIVERGENCE_STRONG_GAP``.

    Returns:
        A dict with ``status`` ("diverge", "agree", "neutral" or "no overlap"),
        ``month``, ``z_a``, ``z_b``, ``gap``, ``level_diverges``, ``trend_diverges``,
        ``trend_a``, ``trend_b`` (3-month z changes or ``None``), ``severity``
        ("strong"/"moderate" or ``None``) and ``streak`` (consecutive months, up to
        the latest, of opposite-sign levels outside the noise band).
    """
    min_z = config.DIVERGENCE_MIN_Z if min_z is None else min_z
    strong_gap = config.DIVERGENCE_STRONG_GAP if strong_gap is None else strong_gap
    both = pd.concat([z_a.rename("a"), z_b.rename("b")], axis=1).dropna().sort_index()
    base = {"status": NO_OVERLAP, "month": None, "z_a": None, "z_b": None, "gap": None,
            "level_diverges": False, "trend_diverges": False, "trend_a": None, "trend_b": None,
            "severity": None, "streak": 0}
    if both.empty:
        return base
    month = both.index.max()
    a, b = float(both.at[month, "a"]), float(both.at[month, "b"])
    earlier = month - pd.DateOffset(months=TREND_LOOKBACK_MONTHS)
    trend_a = trend_b = None
    if earlier in both.index:
        trend_a, trend_b = a - float(both.at[earlier, "a"]), b - float(both.at[earlier, "b"])

    def opposite_outside_band(x: float, y: float) -> bool:
        return x * y < 0 and abs(x) >= min_z and abs(y) >= min_z

    level = opposite_outside_band(a, b)
    trend = trend_a is not None and opposite_outside_band(trend_a, trend_b)
    streak = 0
    for _, row in both.iloc[::-1].iterrows():
        if not opposite_outside_band(float(row["a"]), float(row["b"])):
            break
        streak += 1
    if level or trend:
        status = DIVERGE
    elif abs(a) >= min_z and abs(b) >= min_z:
        status = AGREE
    else:
        status = NEUTRAL
    gap = a - b
    return {"status": status, "month": month, "z_a": a, "z_b": b, "gap": gap,
            "level_diverges": level, "trend_diverges": trend, "trend_a": trend_a, "trend_b": trend_b,
            "severity": ("strong" if abs(gap) >= strong_gap else "moderate") if status == DIVERGE else None,
            "streak": streak if level else 0}


def find_divergences(z_wide: pd.DataFrame, pairs: list[tuple[str, str]], labels: dict[str, str]) -> pd.DataFrame:
    """Compare a list of signal pairs and tabulate the result.

    Args:
        z_wide: Z-scores, one column per ``signal_id`` (``data_loading.wide_series``).
        pairs: ``(signal_id_a, signal_id_b)`` pairs; pairs with a missing column are skipped.
        labels: ``{signal_id: readable name}``.

    Returns:
        One row per compared pair (see ``compare_pair``) plus ``signal_a``, ``signal_b``,
        ``label_a`` and ``label_b``; diverging pairs first, largest gap first.
    """
    rows = []
    for a, b in pairs:
        if a not in z_wide.columns or b not in z_wide.columns:
            continue
        result = compare_pair(z_wide[a], z_wide[b])
        rows.append({"signal_a": a, "signal_b": b, "label_a": labels.get(a, a), "label_b": labels.get(b, b), **result})
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["_rank"] = table["status"].map({DIVERGE: 0, NEUTRAL: 2, AGREE: 1, NO_OVERLAP: 3})
    table["_abs_gap"] = table["gap"].abs().fillna(-1)
    return table.sort_values(["_rank", "_abs_gap"], ascending=[True, False]).drop(columns=["_rank", "_abs_gap"]).reset_index(drop=True)


def cross_source_pairs(z_wide: pd.DataFrame) -> list[tuple[str, str]]:
    """List all pairs of headline signals that have z-scores.

    Args:
        z_wide: Z-scores, one column per ``signal_id``.

    Returns:
        Pairs of different headline signals (one per source).
    """
    present = [s for s in HEADLINE_SIGNALS if s in z_wide.columns and z_wide[s].notna().any()]
    return [(present[i], present[j]) for i in range(len(present)) for j in range(i + 1, len(present))]


def within_retail_pairs(z_wide: pd.DataFrame) -> list[tuple[str, str]]:
    """Pair each MSRS subsector with the total retail series.

    Args:
        z_wide: Z-scores, one column per ``signal_id``.

    Returns:
        ``(subsector, total)`` pairs; empty if the total has no z-scores.
    """
    if REFERENCE_SIGNAL not in z_wide.columns or z_wide[REFERENCE_SIGNAL].isna().all():
        return []
    return [(c, REFERENCE_SIGNAL) for c in z_wide.columns
            if c.startswith("msrs_yoy_") and c != REFERENCE_SIGNAL and z_wide[c].notna().any()]


def _side(z: float) -> str:
    """Say whether a z-score is above or below the baseline.

    Args:
        z: A z-score.

    Returns:
        "above" or "below".
    """
    return "above" if z > 0 else "below"


def _move(delta: float) -> str:
    """Say whether a change is a rise or a fall.

    Args:
        delta: A change in z-score.

    Returns:
        "rose" or "fell".
    """
    return "rose" if delta > 0 else "fell"


def describe_divergence(row: pd.Series) -> str:
    """Write one sentence (or two) about a diverging pair, from its numbers only.

    Args:
        row: One diverging row of ``find_divergences``.

    Returns:
        Plain-language text such as "In Jun 2026, X was above its 12-month baseline (+1.2 standard deviations)
        while Y was below it (-0.6): a moderate gap of 1.8 standard deviations."
    """
    month = pd.Timestamp(row["month"]).strftime("%b %Y")
    parts = []
    if row["level_diverges"]:
        text = (f"In {month}, {row['label_a']} was {_side(row['z_a'])} its 12-month baseline "
                f"({row['z_a']:+.1f} standard deviations) while {row['label_b']} was {_side(row['z_b'])} "
                f"({row['z_b']:+.1f}): a {row['severity']} gap of {abs(row['gap']):.1f} standard deviations.")
        if row["streak"] >= 2:
            text += f" They have pointed in opposite directions for {row['streak']} months in a row."
        parts.append(text)
    if row["trend_diverges"]:
        parts.append(
            f"Over the 3 months to {month}, {row['label_a']} {_move(row['trend_a'])} by {abs(row['trend_a']):.1f} standard "
            f"deviations while {row['label_b']} {_move(row['trend_b'])} by {abs(row['trend_b']):.1f}: the trends point in opposite directions.")
    return " ".join(parts)


def summarize(
    state_name: str,
    cross: pd.DataFrame,
    within: pd.DataFrame,
    sources_without_data: list[str],
    max_listed: int = 5,
) -> list[str]:
    """Write the plain-language summary for tab 3 from the divergence tables.

    Args:
        state_name: Active state, for the wording.
        cross: ``find_divergences`` result for the across-source pairs (may be empty).
        within: ``find_divergences`` result for the subsector-versus-total pairs (may be empty).
        sources_without_data: Readable names of sources that have no data yet.
        max_listed: Most diverging pairs described per group.

    Returns:
        A list of paragraphs (plain text, no markdown tables).
    """
    paragraphs: list[str] = []
    if sources_without_data:
        paragraphs.append("No data yet for: " + "; ".join(sources_without_data) + ". Signals without data are not "
                          "compared and nothing is filled in for them.")

    def group(table: pd.DataFrame, title: str, empty_text: str) -> None:
        judged = table[table["status"] != NO_OVERLAP] if not table.empty else table
        if judged.empty:
            paragraphs.append(empty_text)
            return
        n_div = int((judged["status"] == DIVERGE).sum())
        n_strong = int((judged["severity"] == "strong").sum())
        n_agree = int((judged["status"] == AGREE).sum())
        n_neutral = int((judged["status"] == NEUTRAL).sum())
        month = pd.Timestamp(judged["month"].max()).strftime("%b %Y")
        paragraphs.append(
            f"{title}: {len(judged)} pair(s) compared for {state_name}, latest common month {month}. "
            f"{n_div} diverge ({n_strong} strong), {n_agree} agree, {n_neutral} are neutral "
            f"(at least one signal is within {config.DIVERGENCE_MIN_Z} standard deviations of its own baseline).")
        for _, row in judged[judged["status"] == DIVERGE].head(max_listed).iterrows():
            paragraphs.append(describe_divergence(row))
        if n_div > max_listed:
            paragraphs.append(f"{n_div - max_listed} more diverging pair(s) are in the table below.")

    group(cross, "Across sources",
          "Across sources: no comparison is possible yet, because fewer than two sources have z-scores for "
          f"{state_name} with a month in common.")
    group(within, "Within retail (each MSRS subsector against total retail)",
          "Within retail: no comparison is possible yet (no z-scores for total retail).")
    return paragraphs
