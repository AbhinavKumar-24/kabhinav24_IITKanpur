"""Events, live eligibility and offline evaluability (APPROACH.md §3)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from glucotwin.labels import eligibility, find_events, label_status


def _series(*segments):
    """Build (t, g) from (start, end, value) segments sampled every 5 minutes."""
    t, g = [], []
    for start, end, value in segments:
        pts = np.arange(start, end, 5, dtype=float)
        t.append(pts)
        g.append(np.full(len(pts), float(value)))
    return np.concatenate(t), np.concatenate(g)


def test_event_needs_15_minutes(cfg):
    t, g = _series((0, 60, 150), (60, 80, 200), (80, 200, 150))  # 60..75 above: spans 15 min
    ev, unconf = find_events(t, g, cfg)
    assert len(ev) == 1 and ev["onset"].iloc[0] == 60 and ev["confirm"].iloc[0] == 75
    t2, g2 = _series((0, 60, 150), (60, 75, 200), (75, 200, 150))  # 60..70: only 10 minutes
    ev2, unconf2 = find_events(t2, g2, cfg)
    assert len(ev2) == 0 and len(unconf2) == 0  # ended by a low reading: a real non-event


def test_gap_leaves_run_unconfirmed_and_censored_counts_above(cfg):
    t, g = _series((0, 60, 150), (60, 70, 400), (100, 200, 150))  # 400 = sensor ceiling, then a 35-min gap
    ev, unconf = find_events(t, g, cfg)
    assert len(ev) == 0 and list(unconf) == [60.0]


def test_runs_merge_unless_30_minutes_apart(cfg):
    t, g = _series((0, 40, 150), (40, 60, 200), (60, 80, 150), (80, 100, 200), (100, 300, 150))
    ev, _ = find_events(t, g, cfg)
    assert len(ev) == 1  # 25 minutes between runs: one event
    t2, g2 = _series((0, 40, 150), (40, 60, 200), (60, 100, 150), (100, 120, 200), (120, 300, 150))
    ev2, _ = find_events(t2, g2, cfg)
    assert len(ev2) == 2


def _tl(t, g, src=None):
    return pd.DataFrame({"t": t, "glucose": g, "src": src if src is not None else ["reading"] * len(t), "censored": 0})


def test_eligibility_uses_only_the_past(cfg):
    t, g = _series((0, 120, 150), (120, 140, 200), (140, 300, 150))
    el = eligibility(_tl(t, g), cfg, 5)
    at = dict(zip(t, el, strict=True))
    assert not at[30.0]  # under 60 minutes of history
    assert at[100.0]
    assert not at[125.0]  # current reading above 180
    assert not at[160.0]  # a reading above 180 in the last 30 minutes
    assert at[175.0]


def test_label_evaluability(cfg):
    reading_t = np.arange(0, 400, 5, dtype=float)
    events = pd.DataFrame({"onset": [200.0], "confirm": [215.0], "last": [220.0]})
    y, status = label_status(np.array([100.0, 250.0]), reading_t, events, np.array([]), cfg, 5)
    assert list(status) == ["positive", "negative"] and list(y) == [1.0, 0.0]
    sparse = reading_t[(reading_t < 260) | (reading_t > 340)]  # future window badly covered
    _, status2 = label_status(np.array([250.0]), sparse, events.iloc[:0], np.array([]), cfg, 5)
    assert status2[0] == "unscorable"
    _, status3 = label_status(np.array([100.0]), reading_t, events.iloc[:0], np.array([150.0]), cfg, 5)
    assert status3[0] == "unscorable"  # an unconfirmed run inside the window
