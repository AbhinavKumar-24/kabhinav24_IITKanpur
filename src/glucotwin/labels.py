"""Events, live prediction eligibility and offline label evaluability (APPROACH.md §3).

* Prediction eligibility uses only data at or before t. It decides whether a prediction (and an
  alert) can be made, live or in replay.
* Label evaluability looks after t and is used only in offline evaluation. Windows that cannot be
  scored are "unscorable": excluded and counted, never treated as negative.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .align import window_coverage


def find_events(
    t: np.ndarray, g: np.ndarray, cfg: dict, persistence: float | None = None
) -> tuple[pd.DataFrame, np.ndarray]:
    """Confirmed excursions and unconfirmed above-threshold runs from observed readings.

    An excursion is a run of readings above the threshold spanning at least ``persistence``
    minutes with no gap longer than ``max_gap``. Onset is the first reading of the run and the
    event is confirmed ``persistence`` minutes later. Runs closer than ``merge_min`` merge into
    one event. A short run that ends because of a gap or the end of the data is "unconfirmed".
    Censored readings are unambiguous here: 400 counts as above, 40 as below.
    """
    ev = cfg["event"]
    thr, max_gap, merge = ev["threshold_mgdl"], ev["max_gap_min"], ev["merge_min"]
    persist = ev["persistence_min"] if persistence is None else persistence
    t = np.asarray(t, dtype=float)
    above = np.asarray(g, dtype=float) > thr
    n = len(t)
    confirmed, unconfirmed = [], []
    i = 0
    while i < n:
        if not above[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and above[j + 1] and t[j + 1] - t[j] <= max_gap:
            j += 1
        if t[j] - t[i] >= persist:
            confirmed.append((t[i], t[j]))
        else:
            ended_low = j + 1 < n and t[j + 1] - t[j] <= max_gap and not above[j + 1]
            if not ended_low:
                unconfirmed.append(t[i])
        i = j + 1
    events: list[list[float]] = []
    for start, last in confirmed:
        if events and start - events[-1][2] <= merge:
            events[-1][2] = last
        else:
            events.append([start, start + persist, last])
    return pd.DataFrame(events, columns=["onset", "confirm", "last"]), np.asarray(unconfirmed, dtype=float)


def eligibility(tl: pd.DataFrame, cfg: dict, interval: float) -> np.ndarray:
    """Live rule: may a prediction be made at each timeline row? Uses data at or before t only."""
    thr = cfg["event"]["threshold_mgdl"]
    el = cfg["eligibility"]
    is_reading = (tl["src"] == "reading").to_numpy()
    t = tl["t"].to_numpy(float)
    g = tl["glucose"].to_numpy(float)
    rt = t[is_reading]
    if len(rt) == 0:
        return np.zeros(len(tl), dtype=bool)
    above_t = rt[g[is_reading] > thr]
    recent = np.searchsorted(above_t, t, side="left") - np.searchsorted(
        above_t, t - el["recent_high_window_min"], side="left"
    )
    history = t - rt[0] >= el["history_min"]
    cov = window_coverage(rt, t - el["history_min"], t, interval)
    return is_reading & (g <= thr) & (recent == 0) & history & (cov >= el["history_min_coverage"])


def label_status(
    t_query: np.ndarray,
    reading_t: np.ndarray,
    events: pd.DataFrame,
    unconfirmed: np.ndarray,
    cfg: dict,
    interval: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Offline rule: positive, negative or unscorable for the window (t, t + horizon]."""
    ev = cfg["evaluability"]
    horizon = ev["horizon_min"]
    q = np.asarray(t_query, dtype=float)
    onsets = np.sort(events["onset"].to_numpy(float)) if len(events) else np.array([])
    pos = (np.searchsorted(onsets, q + horizon, side="right") - np.searchsorted(onsets, q, side="right")) > 0
    cov = window_coverage(reading_t, q, q + horizon, interval)
    unconf_sorted = np.sort(unconfirmed)
    unconf = (
        np.searchsorted(unconf_sorted, q + horizon, side="right") - np.searchsorted(unconf_sorted, q, side="right")
    ) > 0
    neg = ~pos & (cov >= ev["min_coverage"]) & ~unconf
    status = np.where(pos, "positive", np.where(neg, "negative", "unscorable"))
    y = np.where(pos, 1.0, np.where(neg, 0.0, np.nan))
    return y, status


def recent_coverage(t_query: np.ndarray, reading_t: np.ndarray, cfg: dict, interval: float) -> np.ndarray:
    """Share of expected readings observed in the last ``recent_window_min`` (alert rule)."""
    w = cfg["alerts"]["recent_window_min"]
    q = np.asarray(t_query, dtype=float)
    return window_coverage(reading_t, q - w, q, interval)
