"""Deterministic alert-to-event matching and the alert engine (APPROACH.md §6.2, §7)."""

from __future__ import annotations

import numpy as np

from glucotwin.alerts import alert_rate, choose_threshold, simulate_alerts
from glucotwin.evaluate import match_alerts


def test_one_alert_is_credited_to_one_event(cfg):
    res = match_alerts(np.array([0.0]), np.array(["positive"]), np.array([50.0, 100.0]), cfg)
    assert res["event_class"] == ["early", "missed"]
    assert res["leads"] == [50.0] and res["alert_class"] == ["true"]


def test_late_false_and_unscorable_alerts(cfg):
    res = match_alerts(
        np.array([90.0, 300.0, 600.0]), np.array(["positive", "negative", "unscorable"]), np.array([100.0]), cfg
    )
    assert res["event_class"] == ["late"]  # only 10 minutes of warning
    assert res["alert_class"] == ["true", "false", "unscorable"]


def test_earliest_alert_sets_the_lead_time(cfg):
    res = match_alerts(np.array([10.0, 70.0]), np.array(["positive", "positive"]), np.array([100.0]), cfg)
    assert res["event_class"] == ["early"] and res["leads"] == [90.0]


def test_cooldown_and_recent_coverage(cfg):
    t = np.arange(0, 205, 5, dtype=float)
    scores = np.ones(len(t))
    cov = np.ones(len(t))
    assert list(t[simulate_alerts(t, scores, cov, 0.5, cfg)]) == [0.0, 90.0, 180.0]
    cov[t == 0] = 0.2  # too few recent readings: no alert at t=0
    assert list(t[simulate_alerts(t, scores, cov, 0.5, cfg)]) == [5.0, 95.0, 185.0]


def test_threshold_respects_alert_budget(cfg):
    rng = np.random.default_rng(1)
    groups = []
    for _ in range(4):
        t = np.arange(0, 2 * 1440, 5, dtype=float)
        groups.append((t, rng.random(len(t)), np.ones(len(t)), 2.0))
    thr = choose_threshold(groups, cfg)
    assert alert_rate(groups, thr, cfg) <= cfg["alerts"]["budget_per_day"]
