"""The personal twin: fitting, censoring, priors and anchoring (APPROACH.md §4)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from glucotwin import twin as tw


def _inputs(meal_t, carbs):
    meal_t = np.asarray(meal_t, dtype=float)
    return tw.TwinInputs(
        meal_t, np.asarray(carbs, float), np.zeros(len(meal_t)), np.zeros(len(meal_t)), np.array([]), np.array([])
    )


def test_fit_recovers_known_parameters():
    spec = tw.TwinSpec(())
    inp = _inputs([60, 360, 720, 1500, 1800, 2160], [60, 80, 50, 70, 40, 90])
    truth = np.array([100.0, math.log(1.2), math.log(50.0)])
    t = np.arange(0, 2600, 5, dtype=float)
    g = tw.predict(truth, spec, inp, t) + np.random.default_rng(0).normal(0, 3, len(t))
    mu, sd = spec.generic_prior()
    est = tw.fit(inp, t, g, spec, mu, sd, sigma=20.0, max_nfev=100)
    r = spec.readable(est)
    assert abs(r["baseline_mgdl"] - 100) < 3
    assert abs(r["rise_per_g_carb"] - 1.2) < 0.1
    assert abs(r["time_to_peak_min"] - 50) < 4


def test_no_data_returns_prior_and_anchor_starts_at_current_glucose():
    spec = tw.TwinSpec(())
    mu, sd = spec.generic_prior()
    inp = _inputs([0], [50])
    theta = tw.fit(inp, np.array([]), np.array([]), spec, mu, sd)
    assert np.allclose(theta, spec.clip(mu))
    f = tw.anchored(theta, spec, inp, np.array([30.0]), np.array([173.0]), np.array([0.0, 60.0]), 45.0)
    assert np.isclose(f[0, 0], 173.0)


def test_fit_data_excludes_censored_readings(pdatas):
    p = next(iter(pdatas.values()))
    p.reading_c = p.reading_c.copy()
    p.reading_c[:5] = 1  # pretend these were sensor-ceiling readings
    t, _ = p.fit_data()
    assert not np.isin(p.reading_t[:5], t).any()
    p.reading_c[:5] = 0


def test_profile_prior_follows_the_profile():
    spec = tw.TwinSpec(())
    rng = np.random.default_rng(3)
    hb = rng.uniform(5.0, 9.0, 20)
    thetas = {f"P{i}": np.array([100.0, math.log(0.4 + 0.3 * (h - 5)), math.log(55.0)]) for i, h in enumerate(hb)}
    prof = pd.DataFrame(
        {"hba1c_pct": hb, "fasting_glucose": 100.0, "homa_ir": 2.0, "bmi": 27.0, "age": 50.0},
        index=list(thetas),
    )
    prior = tw.fit_prior(thetas, prof, spec)
    lo_mu, _ = prior.p1(pd.Series({"hba1c_pct": 5.2, "fasting_glucose": 100, "homa_ir": 2, "bmi": 27, "age": 50}))
    hi_mu, _ = prior.p1(pd.Series({"hba1c_pct": 8.8, "fasting_glucose": 100, "homa_ir": 2, "bmi": 27, "age": 50}))
    assert hi_mu[1] > lo_mu[1]
    again = tw.Prior.from_dict(prior.to_dict())
    assert np.allclose(again.p1(prof.iloc[0])[0], prior.p1(prof.iloc[0])[0])
