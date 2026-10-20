"""Alert engine (APPROACH.md §7). Live rules only: it never sees label evaluability or data after t."""

from __future__ import annotations

import numpy as np


def simulate_alerts(t: np.ndarray, score: np.ndarray, cov30: np.ndarray, threshold: float, cfg: dict) -> np.ndarray:
    """Indices of the moments where an alert fires.

    ``t``, ``score`` and ``cov30`` describe prediction-eligible moments in time order. An alert
    fires when the score reaches the threshold, enough recent readings are present, and no
    alert fired in the cooldown period before.
    """
    al = cfg["alerts"]
    t = np.asarray(t, dtype=float)
    ok = (np.asarray(score) >= threshold) & (np.asarray(cov30) >= al["recent_min_coverage"])
    cand = np.flatnonzero(ok)
    ct = t[cand]
    picks, k = [], 0
    while k < len(cand):
        picks.append(cand[k])
        k = int(np.searchsorted(ct, ct[k] + al["cooldown_min"], side="left"))
    return np.asarray(picks, dtype=int)


def alert_rate(groups: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]], threshold: float, cfg: dict) -> float:
    """Mean alerts per patient-day across participants (t, score, cov30, days)."""
    rates = [len(simulate_alerts(t, s, c, threshold, cfg)) / d for t, s, c, d in groups if d > 0]
    return float(np.mean(rates)) if rates else 0.0


def choose_threshold(groups: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]], cfg: dict) -> float:
    """Lowest threshold whose mean alert rate stays within the daily budget."""
    budget = cfg["alerts"]["budget_per_day"]
    scores = np.concatenate([s for _, s, _, _ in groups]) if groups else np.array([])
    scores = scores[np.isfinite(scores)]
    if len(scores) == 0:
        return float("inf")
    cands = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, 401)))
    if alert_rate(groups, cands[-1], cfg) > budget:
        return float(np.nextafter(cands[-1], np.inf))
    lo, hi = 0, len(cands) - 1  # rate(cands[hi]) <= budget
    while lo < hi:
        mid = (lo + hi) // 2
        if alert_rate(groups, cands[mid], cfg) <= budget:
            hi = mid
        else:
            lo = mid + 1
    return float(cands[lo])


def status(g_now: float, cov30: float, eligible: bool, alert_now: bool, cfg: dict) -> str:
    """What the dashboard shows for the current moment."""
    if np.isnan(g_now):
        return "no reading"
    if g_now > cfg["event"]["threshold_mgdl"]:
        return "above range"
    if cov30 < cfg["alerts"]["recent_min_coverage"]:
        return "insufficient data"
    if not eligible:
        return "not enough history"
    return "alert" if alert_now else "monitoring"
