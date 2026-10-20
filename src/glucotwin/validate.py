"""Validation checks and stop rules for the evidence track (APPROACH.md §2.1).

A failed stop rule raises :class:`ValidationError`. Source-data assumptions are never silently
repaired just to let later steps run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import ValidationError


def minutes(times: pd.Series | pd.DatetimeIndex, origin: pd.Timestamp) -> np.ndarray:
    """Minutes since ``origin`` as floats."""
    delta = pd.to_datetime(times) - origin
    return np.asarray(delta / pd.Timedelta(minutes=1), dtype=float)


def reconstruct_native(
    times: pd.Series, values: pd.Series, interval_min: float, tolerance: float = 0.6
) -> tuple[np.ndarray, str]:
    """Find the rows that hold a sensor's own readings in a 1-minute table.

    CGMacros stores every sensor on 1-minute rows, but Dexcom samples every 5 minutes and Libre
    every 15. The rows in between can be blank, carried forward or interpolated. Returns a
    boolean mask over the rows and the detected mode. Mode "unknown" means the native readings
    could not be identified, which is a stop rule.
    """
    v = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    present = ~np.isnan(v)
    mask = np.zeros(len(v), dtype=bool)
    if present.sum() < 3:
        mask[present] = True
        return mask, "too_few"
    t = minutes(pd.Series(times), pd.Timestamp(pd.Series(times).iloc[0]))
    idx = np.flatnonzero(present)
    tt, vv = t[idx], v[idx]
    step = np.diff(tt)
    if np.median(step) >= 0.8 * interval_min:
        mask[idx] = True
        return mask, "blank"

    if np.mean(np.diff(vv) == 0) >= 0.5:
        # Carried forward: a reading starts each run of equal values; a long run holds several
        # equal readings one interval apart.
        last_t, prev = -np.inf, np.nan
        for k in range(len(idx)):
            if vv[k] != prev or tt[k] - last_t >= interval_min - 0.5:
                mask[idx[k]] = True
                last_t = tt[k]
            prev = vv[k]
        return mask, "carried_forward"

    # Interpolated: readings sit at knots one interval apart. Work per contiguous segment, try
    # every phase and keep the one whose straight lines reproduce all rows.
    seg_breaks = np.flatnonzero(step > 1.5) + 1
    segments = np.split(np.arange(len(idx)), seg_breaks)
    period = int(round(interval_min))
    for seg in segments:
        if len(seg) == 0:
            continue
        st, sv = tt[seg], vv[seg]
        offsets = np.round(st - st[0]).astype(int)
        best = None
        for phase in range(period):
            knots = np.flatnonzero((offsets - phase) % period == 0)
            if len(knots) < 2:
                continue
            recon = np.interp(st, st[knots], sv[knots])
            inside = (st >= st[knots[0]]) & (st <= st[knots[-1]])
            err = np.abs(recon - sv)[inside]
            score = np.quantile(err, 0.99) if len(err) else np.inf
            if best is None or score < best[0]:
                best = (score, knots)
        if best is None:
            if len(seg) <= period:  # a fragment shorter than one interval holds one reading
                mask[idx[seg[0]]] = True
                continue
            return mask, "unknown"
        if best[0] > tolerance:
            return mask, "unknown"
        mask[idx[seg[best[1]]]] = True
    return mask, "interpolated"


def hba1c_groups(
    hba1c: pd.Series, thresholds: tuple[float, float], expected: dict[str, int] | None
) -> tuple[pd.Series, str]:
    """Interpret HbA1c as percent only if the published group sizes reproduce.

    The CGMacros dictionary labels the column mmol/mol while its values (4.6-8.5) and the
    dataset's group thresholds (5.7%, 6.4%) are in percent. Raises ValidationError when the
    percent reading cannot be justified.
    """
    vals = pd.to_numeric(hba1c, errors="coerce")
    if vals.isna().any():
        raise ValidationError(f"HbA1c is missing for {int(vals.isna().sum())} participants.")
    lo_t, hi_t = thresholds
    if not vals.between(3.0, 20.0).all():
        raise ValidationError(
            "HbA1c values fall outside the plausible percent range (3-20); the unit cannot be "
            f"treated as percent. Observed range {vals.min():.1f}-{vals.max():.1f}."
        )
    groups = pd.Series(
        np.where(vals < lo_t, "healthy", np.where(vals <= hi_t, "prediabetes", "t2d")),
        index=hba1c.index,
    )
    counts = groups.value_counts().to_dict()
    detail = (
        "Dictionary unit says mmol/mol; values span "
        f"{vals.min():.1f}-{vals.max():.1f}, which is the percent range, and the dataset groups "
        f"participants at {lo_t}% and {hi_t}%. Groups recomputed as percent: {counts}."
    )
    if expected:
        want = {k: int(v) for k, v in expected.items()}
        got = {k: int(counts.get(k, 0)) for k in want}
        if got != want:
            raise ValidationError(
                f"HbA1c check failed: groups recomputed as percent are {got}, but the dataset "
                f"reports {want}. The unit cannot be confirmed, so the analysis stops."
            )
        detail += f" They match the published counts {want}."
    return groups, detail


def scale_mets(raw: pd.Series, scale: float, plausible: tuple[float, float]) -> pd.Series:
    """METs are stored x10; divide and check the result is plausible."""
    mets = pd.to_numeric(raw, errors="coerce") / scale
    valid = mets.dropna()
    if len(valid):
        lo, hi = valid.quantile(0.01), valid.quantile(0.99)
        if lo < plausible[0] or hi > plausible[1]:
            raise ValidationError(
                f"METs after dividing by {scale} span {lo:.2f}-{hi:.2f} (1st-99th percentile), "
                f"outside the plausible range {plausible}."
            )
    return mets


def censor_flags(glucose: np.ndarray, low: float, high: float) -> np.ndarray:
    """-1 for readings at the sensor floor, +1 at the ceiling, 0 otherwise."""
    flags = np.zeros(len(glucose), dtype=np.int8)
    flags[glucose <= low] = -1
    flags[glucose >= high] = 1
    return flags


def sensor_agreement(primary: pd.DataFrame, other: pd.DataFrame, tolerance_min: float = 2.5) -> float:
    """Mean absolute relative difference (%) between two sensors at matched times."""
    if primary.empty or other.empty:
        return float("nan")
    p = primary.sort_values("time")
    o = other.sort_values("time")
    merged = pd.merge_asof(
        o[["time", "glucose", "censored"]],
        p[["time", "glucose", "censored"]],
        on="time",
        direction="nearest",
        tolerance=pd.Timedelta(minutes=tolerance_min),
        suffixes=("_o", "_p"),
    ).dropna()
    merged = merged[(merged["censored_o"] == 0) & (merged["censored_p"] == 0)]
    if merged.empty:
        return float("nan")
    return float((np.abs(merged["glucose_o"] - merged["glucose_p"]) / merged["glucose_p"]).mean() * 100)


def meal_order_fraction(meals: pd.DataFrame) -> tuple[float, int]:
    """Share of days with breakfast, lunch and dinner logged in that clock order."""
    if meals.empty or "meal_type" not in meals:
        return float("nan"), 0
    m = meals.dropna(subset=["meal_type"]).copy()
    m["kind"] = m["meal_type"].str.lower().str.strip()
    m = m[m["kind"].isin(["breakfast", "lunch", "dinner"])]
    if m.empty:
        return float("nan"), 0
    m["day"] = m["time"].dt.date
    ok = total = 0
    for _, day in m.groupby(["pid", "day"]):
        firsts = day.groupby("kind")["time"].min()
        if {"breakfast", "lunch", "dinner"} <= set(firsts.index):
            total += 1
            hours = [firsts[k].hour + firsts[k].minute / 60 for k in ("breakfast", "lunch", "dinner")]
            ok += int(hours[0] < hours[1] < hours[2])
    return (ok / total if total else float("nan")), total


def require_increasing(times: pd.Series, where: str) -> None:
    """Stop rule: timestamps must strictly increase within a participant file."""
    t = pd.to_datetime(times)
    if t.isna().any():
        raise ValidationError(f"{where}: {int(t.isna().sum())} timestamps could not be parsed.")
    if not t.is_monotonic_increasing or t.duplicated().any():
        raise ValidationError(f"{where}: timestamps do not strictly increase.")
