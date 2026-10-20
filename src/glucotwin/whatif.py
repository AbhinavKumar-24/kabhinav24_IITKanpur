"""User-chosen what-if scenarios on the personal twin (APPROACH.md §9).

The screen shows a baseline and a scenario side by side. There is no optimisation and no
suggested change; outputs are simulated estimates, not measured effects or recommendations.
"""

from __future__ import annotations

import numpy as np

from . import twin as tw
from .features import PData

DISCLAIMER = (
    "Simulated estimate from a simple personal meal-response model - not a physiological "
    "simulation, a measured effect or a treatment recommendation."
)


def _inputs_until(p: PData, now: float) -> tw.TwinInputs:
    inp = p.inputs.window(-np.inf, now)
    keep = inp.act_t <= now
    return tw.TwinInputs(
        inp.meal_t, inp.carbs, inp.other, inp.debt, inp.act_t[keep], inp.act_cum[keep], inp.memory, inp.act_window
    )


def _band_at(band: dict, horizons: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hs = np.array(sorted(int(h) for h in band), dtype=float)
    lo = np.array([band[str(int(h))][0] if str(int(h)) in band else band[int(h)][0] for h in hs])
    hi = np.array([band[str(int(h))][1] if str(int(h)) in band else band[int(h)][1] for h in hs])
    return np.interp(horizons, np.r_[0, hs], np.r_[0, lo]), np.interp(horizons, np.r_[0, hs], np.r_[0, hi])


def scenario(
    p: PData,
    spec: tw.TwinSpec,
    theta: np.ndarray,
    tau_r: float,
    band: dict,
    now: float,
    g_now: float,
    mode: str,
    carbs: float,
    walk_min: float,
    cfg: dict,
) -> dict:
    """Baseline vs scenario for the next ``whatif.horizon_min`` minutes.

    mode "last": change the carbohydrate of the most recent meal (within 3 hours).
    mode "next": add a meal now with ``carbs`` grams of carbohydrate.
    A hypothetical walk is only available when the activity extension (E2) was selected.
    """
    wc = cfg["whatif"]
    thr = cfg["event"]["threshold_mgdl"]
    horizons = np.arange(0, wc["horizon_min"] + 5, 5, dtype=float)
    x = now + horizons
    base_in = _inputs_until(p, now)
    ghat_now = tw.predict(theta, spec, base_in, np.array([now]), np.array([now]))[0]
    resid = g_now - ghat_now
    decay = np.exp(-horizons / tau_r)
    baseline = tw.predict(theta, spec, base_in, x, x) + resid * decay

    meal_t, c, other, debt = base_in.meal_t.copy(), base_in.carbs.copy(), base_in.other.copy(), base_in.debt.copy()
    changed, warnings = [], []
    recent = np.flatnonzero((meal_t <= now) & (meal_t >= now - 180))
    if mode == "last" and len(recent):
        k = recent[-1]
        changed.append(f"Last meal carbohydrate: {c[k]:.0f} g -> {carbs:.0f} g")
        c[k] = carbs
        walk_start = meal_t[k]
    else:
        if mode == "last":
            warnings.append("No meal in the last 3 hours; the scenario adds a meal now instead.")
        changed.append(f"New meal now with {carbs:.0f} g of carbohydrate")
        meal_t, c = np.r_[meal_t, now], np.r_[c, carbs]
        other, debt = np.r_[other, 0.0], np.r_[debt, debt[-1] if len(debt) else 0.0]
        walk_start = now
    act_t, act_cum = base_in.act_t, base_in.act_cum
    if walk_min > 0:
        if spec.has("E2"):
            extra = walk_min * (wc["walk_mets"] - cfg["features"]["met_rest_level"])
            start = max(walk_start, now)
            last = act_cum[-1] if len(act_cum) else 0.0
            act_t = np.r_[act_t, start + walk_min]
            act_cum = np.r_[act_cum, last + extra]
            changed.append(f"Hypothetical walk: {walk_min:.0f} minutes at about {wc['walk_mets']} METs")
        else:
            warnings.append("Walk scenario unavailable: the activity extension (E2) was not selected for this twin.")
    scen_in = tw.TwinInputs(meal_t, c, other, debt, act_t, act_cum, base_in.memory, base_in.act_window)
    scen = tw.predict(theta, spec, scen_in, x, x) + resid * decay

    logged = p.meals[p.meals["t"] <= now]["carbs_eaten"]
    if len(logged) < wc["min_logged_meals"]:
        warnings.append(f"Low evidence: only {len(logged)} logged meals so far.")
    elif not (logged.min() <= carbs <= logged.max()):
        warnings.append(
            f"Low evidence: {carbs:.0f} g is outside this person's logged range ({logged.min():.0f}-{logged.max():.0f} g)."
        )
    lo, hi = _band_at(band, horizons)

    def summary(curve):
        return {"peak": float(curve.max()), "minutes_above": float(5 * (curve > thr).sum())}

    return {
        "horizons": horizons,
        "baseline": baseline,
        "scenario": scen,
        "band_lo": lo,
        "band_hi": hi,
        "baseline_summary": summary(baseline),
        "scenario_summary": summary(scen),
        "changed": changed,
        "warnings": warnings,
        "disclaimer": DISCLAIMER,
    }
