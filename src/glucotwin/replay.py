"""Historical replay for the dashboard (APPROACH.md §10).

``precompute`` + ``state_at`` is the fast path the dashboard uses: features, scores and alerts are
computed causally once, then filtered at the replay time. ``replay_state`` deletes all data
after the replay time first; ``tests/test_replay.py`` checks both paths agree.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import twin as tw
from .alerts import simulate_alerts, status
from .explain import reasons
from .features import PData, daily_params, prepare_participant, twin_features
from .models import Fitted, score
from .schema import TrackData
from .whatif import _band_at


@dataclass
class Precomputed:
    pdata: PData
    spec: tw.TwinSpec
    params: np.ndarray
    X: pd.DataFrame
    scores: np.ndarray
    alerts: np.ndarray
    model: Fitted


def precompute(
    pdata: PData, bundle: dict, fitted: dict[str, Fitted], cfg: dict, params: np.ndarray | None = None
) -> Precomputed:
    spec = tw.TwinSpec(tuple(bundle["spec"]))
    prior = tw.Prior.from_dict(bundle["prior"])
    if params is None or len(params) < len(pdata.day_starts):
        params = daily_params(pdata, spec, *prior.p1(pdata.profile), cfg)
    params = np.asarray(params)[: len(pdata.day_starts)]
    tfeat = twin_features(pdata, spec, params, bundle["tau_r"])
    X = pd.concat([pdata.moments, tfeat], axis=1)
    model = fitted[bundle["selected_model"]]
    s = score(model, X) if len(X) else np.array([])
    m = pdata.moments
    alerts = (
        simulate_alerts(
            m["t"].to_numpy(float), s, m["cov30"].to_numpy(float), bundle["thresholds"][bundle["selected_model"]], cfg
        )
        if len(X)
        else np.array([], int)
    )
    return Precomputed(pdata, spec, params, X, s, alerts, model)


def state_at(pre: Precomputed, now: float, bundle: dict, cfg: dict, with_reasons: bool = True) -> dict:
    """Everything the dashboard shows at replay time ``now`` (minutes since the participant's start)."""
    p = pre.pdata
    rt = p.reading_t
    k = np.searchsorted(rt, now, side="right") - 1
    g_now = float(p.reading_g[k]) if k >= 0 else float("nan")
    t_last = float(rt[k]) if k >= 0 else float("-inf")
    m = p.moments
    mt = m["t"].to_numpy(float)
    j = np.flatnonzero(mt == t_last)
    eligible = bool(len(j)) and (now - t_last) <= p.interval
    risk = float(pre.scores[j[0]]) if eligible else float("nan")
    cov30 = float(np.clip(((rt > now - 30) & (rt <= now)).sum() / (30.0 / p.interval), 0, 1))
    alert_t = mt[pre.alerts] if len(pre.alerts) else np.array([])
    alerts_so_far = alert_t[alert_t <= now]
    alert_now = eligible and len(alerts_so_far) > 0 and alerts_so_far[-1] == t_last
    st = status(g_now if (now - t_last) <= p.interval else float("nan"), cov30, eligible, alert_now, cfg)
    day = int(np.searchsorted(p.day_starts, now, side="right") - 1)
    horizons = np.arange(0, cfg["whatif"]["horizon_min"] + 5, 5, dtype=float)
    forecast = lo = hi = None
    if k >= 0 and day >= 0:
        forecast = tw.anchored(
            pre.params[day], pre.spec, p.inputs, np.array([now]), np.array([g_now]), horizons, bundle["tau_r"]
        )[0]
        blo, bhi = _band_at(bundle["band"], horizons)
        lo, hi = forecast + blo, forecast + bhi
    tl = p.tl[p.tl["t"] <= t_last]
    confirmed = p.events[p.events["confirm"] <= now] if len(p.events) else p.events
    out = {
        "now": now,
        "time": p.t0 + pd.Timedelta(minutes=now),
        "g_now": g_now,
        "risk": risk,
        "status": st,
        "cov30": cov30,
        "alerts_so_far": alerts_so_far,
        "alert_now": bool(alert_now),
        "horizons": horizons,
        "forecast": forecast,
        "band_lo": lo,
        "band_hi": hi,
        "trace": tl,
        "events": confirmed,
        "meals": p.meals[p.meals["t"] <= now],
        "activity": p.activity[p.activity["t"] <= now],
        "params": pre.spec.readable(pre.params[day]) if day >= 0 else {},
        "reasons": [],
    }
    if with_reasons and eligible:
        out["reasons"] = reasons(pre.model, pre.X.iloc[[j[0]]])
    return out


def replay_state(
    td: TrackData, pid: str, until: pd.Timestamp, bundle: dict, fitted: dict[str, Fitted], cfg: dict
) -> dict:
    """Delete everything after ``until``, then compute the state from scratch."""
    sub = td.truncated(pid, until)
    p = prepare_participant(sub, pid, cfg)
    pre = precompute(p, bundle, fitted, cfg)
    now = float((until - p.t0) / pd.Timedelta(minutes=1))
    return state_at(pre, now, bundle, cfg)
