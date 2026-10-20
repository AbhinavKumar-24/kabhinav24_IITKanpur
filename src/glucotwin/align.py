"""Alignment on native readings, interval-aware coverage and the heuristic sleep estimate.

There is no fixed clock grid: each participant's timeline is anchored to the primary sensor's
own reading times. Coverage always compares readings present with readings expected at that
sensor's native interval, so a regular 15-minute Libre series has 100% coverage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import SLEEP_COLUMNS


def wear_coverage(times: pd.Series, interval_min: float) -> float:
    """Readings present / readings expected between the first and last reading."""
    t = pd.to_datetime(pd.Series(times)).sort_values()
    if len(t) < 2:
        return 0.0
    span = (t.iloc[-1] - t.iloc[0]) / pd.Timedelta(minutes=1)
    expected = span / interval_min + 1
    return float(min(len(t) / expected, 1.0))


def window_counts(t_obs: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Number of observations with start < t <= end, for each window."""
    return np.searchsorted(t_obs, ends, side="right") - np.searchsorted(t_obs, starts, side="right")


def window_coverage(t_obs: np.ndarray, starts: np.ndarray, ends: np.ndarray, interval_min: float) -> np.ndarray:
    expected = (np.asarray(ends) - np.asarray(starts)) / interval_min
    with np.errstate(divide="ignore", invalid="ignore"):
        cov = window_counts(t_obs, starts, ends) / expected
    return np.clip(np.nan_to_num(cov), 0.0, 1.0)


def build_timeline(readings: pd.DataFrame, interval_min: float, max_gap_min: float, t0: pd.Timestamp) -> pd.DataFrame:
    """Readings plus imputed rows for short gaps.

    A gap between two readings that is at most ``max_gap_min`` long gets linearly interpolated
    rows at the expected sampling times, tagged ``imputed``. Longer gaps stay empty.
    Imputed rows are for features only; labels never use them.
    """
    r = readings.sort_values("time")
    t = np.asarray((r["time"] - t0) / pd.Timedelta(minutes=1), dtype=float)
    g = r["glucose"].to_numpy(float)
    c = r["censored"].to_numpy(np.int8)
    rows_t, rows_g, rows_src, rows_c = [t], [g], [np.full(len(t), "reading", dtype=object)], [c]
    if len(t) > 1:
        gaps = np.diff(t)
        for i in np.flatnonzero((gaps > 1.5 * interval_min) & (gaps <= max_gap_min)):
            n_missing = int(round(gaps[i] / interval_min)) - 1
            if n_missing < 1:
                continue
            new_t = t[i] + gaps[i] * np.arange(1, n_missing + 1) / (n_missing + 1)
            rows_t.append(new_t)
            rows_g.append(np.interp(new_t, [t[i], t[i + 1]], [g[i], g[i + 1]]))
            rows_src.append(np.full(n_missing, "imputed", dtype=object))
            rows_c.append(np.zeros(n_missing, dtype=np.int8))
    tl = pd.DataFrame(
        {
            "t": np.concatenate(rows_t),
            "glucose": np.concatenate(rows_g),
            "src": np.concatenate(rows_src),
            "censored": np.concatenate(rows_c),
        }
    )
    tl = tl.sort_values("t", kind="mergesort").reset_index(drop=True)
    tl["time"] = t0 + pd.to_timedelta(tl["t"], unit="m")
    tl["day"] = tl["time"].dt.normalize()
    return tl[["time", "t", "day", "glucose", "src", "censored"]]


def estimate_sleep(activity: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Heuristic overnight sleep estimate from 1-minute heart rate and METs (derived, not measured).

    For each night between ``window_start_hour`` and ``window_end_hour``, sleep is the longest
    block of low activity and below-median heart rate, allowing short interruptions. A night is
    estimated only if heart rate covers enough of the window. The estimate becomes available at
    the end of the window, so nothing later in the day can leak into it.
    """
    sc = cfg["sleep"]
    if activity.empty or activity["hr"].notna().sum() == 0:
        return pd.DataFrame(columns=SLEEP_COLUMNS)
    a = activity.dropna(subset=["hr"]).sort_values("time")
    pid = a["pid"].iloc[0]
    hr_median = a["hr"].median()
    rest_level = cfg["features"]["met_rest_level"]
    start_h, end_h = sc["window_start_hour"], sc["window_end_hour"]
    window_min = ((24 - start_h) + end_h) * 60
    out = []
    first_day = a["time"].min().normalize() - pd.Timedelta(days=1)
    last_day = a["time"].max().normalize()
    for day in pd.date_range(first_day, last_day, freq="D"):
        w_start = day + pd.Timedelta(hours=start_h)
        w_end = w_start + pd.Timedelta(minutes=window_min)
        w = a[(a["time"] > w_start) & (a["time"] <= w_end)]
        coverage = len(w) / window_min
        if coverage < sc["min_coverage"]:
            continue
        rest = ((w["mets"].fillna(rest_level) < rest_level) & (w["hr"] < hr_median)).to_numpy()
        best = run = interrupt = 0
        for is_rest in rest:
            if is_rest:
                run += 1 + interrupt
                interrupt = 0
            elif run > 0 and interrupt < sc["max_interruption_min"]:
                interrupt += 1
            else:
                run = interrupt = 0
            best = max(best, run)
        out.append(
            {
                "pid": pid,
                "night": (w_end.normalize()),
                "sleep_hours": best / 60.0,
                "coverage": coverage,
                "available_time": w_end,
                "src": "derived-heuristic",
                "rmssd": np.nan,
            }
        )
    return pd.DataFrame(out, columns=SLEEP_COLUMNS)
