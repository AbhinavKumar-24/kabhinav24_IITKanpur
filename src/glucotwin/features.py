"""Participant preparation and leakage-safe features (APPROACH.md §5.2).

Every feature at time t uses only data stamped at or before t. ``tests/test_no_leakage.py``
deletes all data after t and checks that every feature at t is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import twin as tw
from .align import build_timeline, window_counts
from .labels import eligibility, find_events, label_status, recent_coverage
from .schema import PROFILE_FEATURES, TrackData

GLUCOSE_FEATURES = ["g_now", "slope_15", "slope_30", "slope_60", "sd_60", "max_180", "mins_since_high"]
MEAL_FEATURES = ["mins_since_meal", "last_meal_carbs", "carbs_180", "carbs_on_board"]
ACTIVITY_FEATURES = ["mets_30", "met_min_since_meal", "hr_30"]
TIME_FEATURES = ["hour_sin", "hour_cos"]
TWIN_FEATURES = ["twin_f60", "twin_f120", "twin_peak", "twin_resid30"]
FAMILIES = {
    "glucose": GLUCOSE_FEATURES,
    "meals": MEAL_FEATURES,
    "activity": ACTIVITY_FEATURES,
    "time": TIME_FEATURES,
    "profile": PROFILE_FEATURES,
    "twin": TWIN_FEATURES,
}


def model_features(model: str, clock_ok: bool) -> list[str]:
    dynamic = GLUCOSE_FEATURES + MEAL_FEATURES + ACTIVITY_FEATURES + (TIME_FEATURES if clock_ok else [])
    if model == "M1":
        return dynamic
    if model == "M2":
        return dynamic + PROFILE_FEATURES
    if model == "M3":
        return dynamic + PROFILE_FEATURES + TWIN_FEATURES
    return ["g_now", "slope_30"]


@dataclass
class PData:
    """One participant, prepared for evaluation or replay."""

    pid: str
    group: str
    profile: pd.Series
    interval: float
    t0: pd.Timestamp
    tl: pd.DataFrame
    reading_t: np.ndarray
    reading_g: np.ndarray
    reading_c: np.ndarray
    meals: pd.DataFrame
    activity: pd.DataFrame
    inputs: tw.TwinInputs
    events: pd.DataFrame
    unconfirmed: np.ndarray
    moments: pd.DataFrame  # eligible prediction moments with labels and base features
    day_starts: np.ndarray
    days: float
    meta: dict = field(default_factory=dict)

    def fit_data(self, before: float = np.inf, after: float = -np.inf) -> tuple[np.ndarray, np.ndarray]:
        """Readings the twin may fit: observed, not censored, in (after, before)."""
        ok = (self.reading_c == 0) & (self.reading_t < before) & (self.reading_t > after)
        return self.reading_t[ok], self.reading_g[ok]

    def targets(self, x: np.ndarray) -> np.ndarray:
        """Observed, non-censored reading nearest to each time in ``x`` (NaN if none close)."""
        rt = self.reading_t
        if len(rt) == 0:
            return np.full(len(x), np.nan)
        idx = np.clip(np.searchsorted(rt, x), 1, len(rt) - 1)
        left, right = idx - 1, idx
        pick = np.where(np.abs(rt[left] - x) <= np.abs(rt[right] - x), left, right)
        ok = (np.abs(rt[pick] - x) <= self.interval / 2 + 0.5) & (self.reading_c[pick] == 0)
        return np.where(ok, self.reading_g[pick], np.nan)


def _prefix(values: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(values)])


def _take(arr: np.ndarray, idx: np.ndarray, valid: np.ndarray, fill: float) -> np.ndarray:
    """arr[idx] where valid, else fill; safe for empty arrays."""
    if len(arr) == 0:
        return np.full(len(idx), fill, dtype=float)
    return np.where(valid, arr[np.clip(idx, 0, len(arr) - 1)], fill)


def glucose_features(tl: pd.DataFrame, cfg: dict, use_imputed: bool = True) -> pd.DataFrame:
    """Rolling glucose features at every timeline row, from rows at or before that row."""
    thr = cfg["event"]["threshold_mgdl"]
    cap = cfg["features"]["cap_min"]
    src = tl["src"].to_numpy()
    rows = np.ones(len(tl), bool) if use_imputed else src == "reading"
    T = tl["t"].to_numpy(float)
    G = tl["glucose"].to_numpy(float)
    Tu, Gu = T[rows], G[rows]
    x = Tu - (Tu[0] if len(Tu) else 0.0)
    P = {
        k: _prefix(v)
        for k, v in {"1": np.ones_like(x), "x": x, "y": Gu, "xy": x * Gu, "xx": x * x, "yy": Gu * Gu}.items()
    }
    out = pd.DataFrame(index=tl.index)
    hi = np.searchsorted(Tu, T, side="right")
    for w in (15, 30, 60):
        lo = np.searchsorted(Tu, T - w, side="right")
        n = P["1"][hi] - P["1"][lo]
        sx, sy = P["x"][hi] - P["x"][lo], P["y"][hi] - P["y"][lo]
        sxy, sxx = P["xy"][hi] - P["xy"][lo], P["xx"][hi] - P["xx"][lo]
        den = n * sxx - sx * sx
        with np.errstate(divide="ignore", invalid="ignore"):
            slope = (n * sxy - sx * sy) / den
        out[f"slope_{w}"] = np.where((n >= 3) & (den > 1e-9), slope, np.nan)
        if w == 60:
            syy = P["yy"][hi] - P["yy"][lo]
            with np.errstate(divide="ignore", invalid="ignore"):
                var = (syy - sy * sy / n) / (n - 1)
            out["sd_60"] = np.where(n >= 2, np.sqrt(np.maximum(var, 0)), np.nan)
    series = pd.Series(Gu, index=pd.to_datetime(Tu, unit="m"))
    roll_max = series.rolling("180min", closed="right").max().to_numpy()
    idx = hi - 1
    out["max_180"] = np.where(idx >= 0, roll_max[np.maximum(idx, 0)], np.nan)
    out["g_now"] = G
    reading = src == "reading"
    above_t = T[reading & (G > thr)]
    j = np.searchsorted(above_t, T, side="right") - 1
    since = T - _take(above_t, j, j >= 0, np.nan)
    out["mins_since_high"] = np.minimum(np.nan_to_num(since, nan=cap), cap)
    return out


def meal_features(q: np.ndarray, meals: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    cap = cfg["features"]["cap_min"]
    decay = cfg["features"]["meal_decay_min"]
    mt = meals["t"].to_numpy(float)
    mc = meals["carbs_eaten"].to_numpy(float)
    out = pd.DataFrame(index=np.arange(len(q)))
    j = np.searchsorted(mt, q, side="right") - 1
    has = j >= 0
    last_t = _take(mt, j, has, np.nan)
    since = np.nan_to_num(q - last_t, nan=cap)
    out["mins_since_meal"] = np.minimum(since, cap)
    out["last_meal_carbs"] = _take(mc, j, has & (since <= cap), 0.0)
    pc = _prefix(mc)
    out["carbs_180"] = pc[np.searchsorted(mt, q, side="right")] - pc[np.searchsorted(mt, q - 180, side="right")]
    if len(mt):
        D = q[:, None] - mt[None, :]
        w = np.where((D >= 0) & (D <= cap), np.exp(-np.clip(D, 0, None) / decay), 0.0)
        out["carbs_on_board"] = w @ mc
    else:
        out["carbs_on_board"] = 0.0
    out["_last_meal_t"] = last_t
    return out


def activity_features(q: np.ndarray, act: pd.DataFrame, last_meal_t: np.ndarray, cfg: dict) -> pd.DataFrame:
    cap = cfg["features"]["cap_min"]
    rest = cfg["features"]["met_rest_level"]
    at = act["t"].to_numpy(float)
    mets = act["mets"].to_numpy(float)
    hr = act["hr"].to_numpy(float)
    dur = act["dur_min"].to_numpy(float)
    m_ok = ~np.isnan(mets)
    h_ok = ~np.isnan(hr)
    pm = _prefix(np.where(m_ok, mets * dur, 0.0))
    pd_ = _prefix(np.where(m_ok, dur, 0.0))
    pe = _prefix(np.where(m_ok, np.maximum(mets - rest, 0.0) * dur, 0.0))
    ph = _prefix(np.where(h_ok, hr, 0.0))
    phn = _prefix(h_ok.astype(float))
    hi = np.searchsorted(at, q, side="right")
    lo = np.searchsorted(at, q - 30, side="right")
    out = pd.DataFrame(index=np.arange(len(q)))
    with np.errstate(divide="ignore", invalid="ignore"):
        out["mets_30"] = np.where(pd_[hi] - pd_[lo] > 0, (pm[hi] - pm[lo]) / (pd_[hi] - pd_[lo]), np.nan)
        out["hr_30"] = np.where(phn[hi] - phn[lo] > 0, (ph[hi] - ph[lo]) / (phn[hi] - phn[lo]), np.nan)
    start = np.where(np.isnan(last_meal_t), q - cap, np.maximum(last_meal_t, q - cap))
    out["met_min_since_meal"] = pe[hi] - pe[np.searchsorted(at, start, side="right")]
    return out


def base_features(pdata: PData, rows: np.ndarray, cfg: dict, use_imputed: bool = True) -> pd.DataFrame:
    """Glucose, meal, activity, time and profile features at timeline rows ``rows``."""
    tl = pdata.tl
    g = glucose_features(tl, cfg, use_imputed).iloc[rows].reset_index(drop=True)
    q = tl["t"].to_numpy(float)[rows]
    m = meal_features(q, pdata.meals, cfg)
    a = activity_features(q, pdata.activity, m["_last_meal_t"].to_numpy(), cfg)
    times = tl["time"].iloc[rows]
    hour = (times.dt.hour + times.dt.minute / 60.0).to_numpy()
    out = pd.concat([g, m.drop(columns="_last_meal_t"), a], axis=1)
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    for f in PROFILE_FEATURES:
        out[f] = float(pdata.profile.get(f, np.nan))
    return out


def twin_inputs(
    meals: pd.DataFrame, activity: pd.DataFrame, sleep: pd.DataFrame, t0: pd.Timestamp, cfg: dict
) -> tw.TwinInputs:
    rest = cfg["features"]["met_rest_level"]
    ref = cfg["sleep"]["reference_hours"]
    mt = meals["t"].to_numpy(float)
    debt = np.zeros(len(mt))
    if len(sleep) and len(mt):
        s = sleep.sort_values("available_time")
        avail = np.asarray((s["available_time"] - t0) / pd.Timedelta(minutes=1), dtype=float)
        k = np.searchsorted(avail, mt, side="right") - 1
        hours = s["sleep_hours"].to_numpy(float)
        debt = np.where(k >= 0, np.maximum(ref - hours[np.maximum(k, 0)], 0.0), 0.0)
    at = activity["t"].to_numpy(float)
    excess = np.maximum(activity["mets"].fillna(rest).to_numpy(float) - rest, 0.0) * activity["dur_min"].to_numpy(float)
    other = (meals[["protein_eaten", "fat_eaten", "fiber_eaten"]].fillna(0).sum(axis=1)).to_numpy(float)
    return tw.TwinInputs(
        mt,
        meals["carbs_eaten"].to_numpy(float),
        other,
        debt,
        at,
        np.cumsum(excess),
        float(cfg["twin"]["meal_memory_min"]),
        float(cfg["twin"]["activity_window_min"]),
    )


def prepare_participant(
    td: TrackData,
    pid: str,
    cfg: dict,
    sensor: str | None = None,
    use_imputed: bool = True,
    persistence: float | None = None,
    exclude_censored: bool = False,
) -> PData | None:
    sensor = sensor or cfg["sensors"]["primary"]
    interval = float(cfg["sensors"][f"{sensor}_interval_min"])
    readings = td.cgm[(td.cgm["pid"] == pid) & (td.cgm["sensor"] == sensor)].sort_values("time")
    readings = readings.dropna(subset=["glucose"])
    if len(readings) < 12:
        return None
    t0 = readings["time"].iloc[0]
    tl = build_timeline(readings, interval, cfg["sensors"]["max_imputation_gap_min"], t0)
    is_r = (tl["src"] == "reading").to_numpy()
    rt, rg = tl["t"].to_numpy(float)[is_r], tl["glucose"].to_numpy(float)[is_r]
    rc = tl["censored"].to_numpy(np.int8)[is_r]
    events, unconfirmed = find_events(rt, rg, cfg, persistence)

    def rel(df: pd.DataFrame) -> pd.DataFrame:
        out = df[df["pid"] == pid].sort_values("time").copy()
        out["t"] = np.asarray((out["time"] - t0) / pd.Timedelta(minutes=1), dtype=float)
        return out.reset_index(drop=True)

    meals = rel(td.meals)
    activity = rel(td.activity)
    sleep = td.sleep[td.sleep["pid"] == pid]
    inputs = twin_inputs(meals, activity, sleep, t0, cfg)
    elig = eligibility(tl, cfg, interval)
    rows = np.flatnonzero(elig)
    q = tl["t"].to_numpy(float)[rows]
    y, status = label_status(q, rt, events, unconfirmed, cfg, interval)
    day_starts = np.asarray((tl["day"].drop_duplicates() - t0) / pd.Timedelta(minutes=1), dtype=float)
    profile = td.profiles.loc[pid]
    pdata = PData(
        pid=pid,
        group=str(profile.get("group", "")),
        profile=profile,
        interval=interval,
        t0=t0,
        tl=tl,
        reading_t=rt,
        reading_g=rg,
        reading_c=rc,
        meals=meals,
        activity=activity,
        inputs=inputs,
        events=events,
        unconfirmed=unconfirmed,
        moments=pd.DataFrame(),
        day_starts=day_starts,
        days=float((rt[-1] - rt[0]) / 1440.0),
    )
    moments = pd.DataFrame(
        {
            "row": rows,
            "t": q,
            "time": tl["time"].to_numpy()[rows],
            "day_idx": np.searchsorted(day_starts, q, side="right") - 1,
            "status": status,
            "y": y,
            "cov30": recent_coverage(q, rt, cfg, interval),
        }
    )
    if exclude_censored and len(moments):
        cens_t = rt[rc != 0]
        near = window_counts(cens_t, q - 60, q + cfg["evaluability"]["horizon_min"]) > 0
        moments = moments[~near].reset_index(drop=True)
        rows = moments["row"].to_numpy()
    base = base_features(pdata, rows, cfg, use_imputed)
    pdata.moments = pd.concat([moments.reset_index(drop=True), base], axis=1)
    return pdata


def prepare_track(td: TrackData, cfg: dict, pids: list[str] | None = None, **variant) -> dict[str, PData]:
    out = {}
    for pid in pids if pids is not None else td.included:
        p = prepare_participant(td, pid, cfg, **variant)
        if p is not None:
            out[pid] = p
    return out


# ------------------------------------------------------------------------------ twin features


def daily_params(pdata: PData, spec: tw.TwinSpec, mu: np.ndarray, sd: np.ndarray, cfg: dict) -> np.ndarray:
    """Twin parameters per day, each fitted on data before that day's midnight (array days x params)."""
    sigma = cfg["twin"]["sigma_mgdl"]
    nfev = cfg["twin"]["max_nfev"]
    out, prev = [], None
    for d0 in pdata.day_starts:
        t_fit, g_fit = pdata.fit_data(before=d0)
        if len(t_fit) == 0:
            theta = spec.clip(mu)
        else:
            theta = tw.fit(pdata.inputs.window(-np.inf, d0), t_fit, g_fit, spec, mu, sd, sigma, x0=prev, max_nfev=nfev)
        out.append(theta)
        prev = theta
    return np.vstack(out)


def twin_curve_at_readings(pdata: PData, spec: tw.TwinSpec, params: np.ndarray) -> np.ndarray:
    """G_hat at every reading, using that reading's own day parameters and data up to it."""
    rt = pdata.reading_t
    day = np.searchsorted(pdata.day_starts, rt, side="right") - 1
    ghat = np.full(len(rt), np.nan)
    for d in np.unique(day):
        sel = day == d
        ghat[sel] = tw.predict(params[d], spec, pdata.inputs, rt[sel], rt[sel])
    return ghat


def twin_features(pdata: PData, spec: tw.TwinSpec, params: np.ndarray, tau_r: float) -> pd.DataFrame:
    """Twin features at each eligible moment (APPROACH.md §5.2, M3)."""
    m = pdata.moments
    out = pd.DataFrame(index=m.index, columns=TWIN_FEATURES, dtype=float)
    if m.empty:
        return out
    ghat_r = twin_curve_at_readings(pdata, spec, params)
    resid = np.where(pdata.reading_c == 0, pdata.reading_g - ghat_r, np.nan)
    ok = ~np.isnan(resid)
    pr, pn = _prefix(np.where(ok, resid, 0.0)), _prefix(ok.astype(float))
    q = m["t"].to_numpy(float)
    hi = np.searchsorted(pdata.reading_t, q, side="right")
    lo = np.searchsorted(pdata.reading_t, q - 30, side="right")
    with np.errstate(divide="ignore", invalid="ignore"):
        out["twin_resid30"] = np.where(pn[hi] - pn[lo] > 0, (pr[hi] - pr[lo]) / (pn[hi] - pn[lo]), np.nan)
    horizons = np.arange(0, 125, 5, dtype=float)
    g_now = m["g_now"].to_numpy(float)
    for d in np.unique(m["day_idx"]):
        sel = (m["day_idx"] == d).to_numpy()
        f = tw.anchored(params[d], spec, pdata.inputs, q[sel], g_now[sel], horizons, tau_r)
        out.loc[sel, "twin_f60"] = f[:, 12]
        out.loc[sel, "twin_f120"] = f[:, 24]
        out.loc[sel, "twin_peak"] = f.max(axis=1)
    return out.astype(float)
