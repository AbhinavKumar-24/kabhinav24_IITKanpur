"""Nested participant-wise evaluation (APPROACH.md §6).

Outer folds are only for reporting. Everything learned - twin priors, tau_r, uncertainty bands,
alert thresholds, extension and model selection - is learned inside the outer training set, with
inner cross-validation where a choice is made. Every model and extension is reported.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from . import twin as tw
from .alerts import choose_threshold, simulate_alerts
from .features import PData, daily_params, twin_features
from .models import MODELS, Fitted, score, train
from .schema import PROFILE_FEATURES

# --------------------------------------------------------------------------- small helpers


def participant_folds(pids: list[str], groups: list[str], k: int, seed: int) -> list[tuple[list[str], list[str]]]:
    k = max(2, min(k, len(pids)))
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    out = []
    for tr, te in sgkf.split(np.zeros(len(pids)), groups, pids):
        out.append(([pids[i] for i in tr], [pids[i] for i in te]))
    return out


def boot(pids: list[str], stat, B: int, seed: int) -> list[float]:
    """Estimate and 95% interval from resampling participants with replacement."""
    if not pids:
        return [float("nan")] * 3
    est = stat(pids)
    rng = np.random.default_rng(seed)
    vals = [stat(list(rng.choice(pids, len(pids), replace=True))) for _ in range(B)]
    vals = np.asarray(vals, dtype=float)
    if np.all(np.isnan(vals)):
        return [float(est), float("nan"), float("nan")]
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return [float(est), float(lo), float(hi)]


def clarke_zone(ref: float, pred: float) -> str:
    """Clarke error grid zone (Clarke et al., Diabetes Care 1987)."""
    if (ref <= 70 and pred <= 70) or (0.8 * ref <= pred <= 1.2 * ref):
        return "A"
    if (ref >= 180 and pred <= 70) or (ref <= 70 and pred >= 180):
        return "E"
    if (70 <= ref <= 290 and pred >= ref + 110) or (130 <= ref <= 180 and pred <= (7 / 5) * ref - 182):
        return "C"
    if (
        (ref >= 240 and 70 <= pred <= 180)
        or (ref <= 175 / 3 and 70 <= pred <= 180)
        or (175 / 3 <= ref <= 70 and pred >= (6 / 5) * ref)
    ):
        return "D"
    return "B"


def profiles_frame(pdatas: dict[str, PData]) -> pd.DataFrame:
    df = pd.DataFrame({pid: p.profile for pid, p in pdatas.items()}).T
    for col in PROFILE_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce") if col in df else np.nan
    return df


def stage1(p: PData, spec: tw.TwinSpec, cfg: dict, cache: dict) -> np.ndarray:
    """Each participant fitted on all their own data with a weak generic prior (prior learning)."""
    key = (p.pid, spec.label)
    if key not in cache:
        mu, sd = spec.generic_prior()
        t, g = p.fit_data()
        cache[key] = tw.fit(
            p.inputs, t, g, spec, mu, sd, cfg["twin"]["sigma_mgdl"], max_nfev=2 * cfg["twin"]["max_nfev"]
        )
    return cache[key]


def prior_for(pids: list[str], pdatas: dict[str, PData], spec: tw.TwinSpec, cfg: dict, cache: dict) -> tw.Prior:
    thetas = {pid: stage1(pdatas[pid], spec, cfg, cache) for pid in pids}
    return tw.fit_prior(thetas, profiles_frame({pid: pdatas[pid] for pid in pids}), spec)


def sampled_moments(p: PData, step: float, t_from: float = -np.inf) -> np.ndarray:
    m = p.moments
    t = m["t"].to_numpy(float)
    sel = np.flatnonzero(t >= t_from)
    if len(sel) == 0:
        return sel
    bucket = np.floor(t[sel] / step)
    first = np.concatenate([[True], bucket[1:] != bucket[:-1]])
    return sel[first]


def forecast_parts(p: PData, spec: tw.TwinSpec, params: np.ndarray, idx: np.ndarray, horizons: np.ndarray):
    """For moments ``idx``: G_hat(t+h) - target and the current residual, so any tau_r can be scored."""
    m = p.moments.iloc[idx]
    q = m["t"].to_numpy(float)
    g_now = m["g_now"].to_numpy(float)
    days = m["day_idx"].to_numpy(int) if params.ndim == 2 else np.zeros(len(q), int)
    base = np.full((len(q), len(horizons)), np.nan)
    resid = np.full(len(q), np.nan)
    for d in np.unique(days):
        sel = days == d
        theta = params[d] if params.ndim == 2 else params
        ghat_now = tw.predict(theta, spec, p.inputs, q[sel], q[sel])
        x = (q[sel][:, None] + horizons[None, :]).ravel()
        ghat = tw.predict(theta, spec, p.inputs, x, np.repeat(q[sel], len(horizons))).reshape(-1, len(horizons))
        base[sel] = ghat
        resid[sel] = g_now[sel] - ghat_now
    target = p.targets((q[:, None] + horizons[None, :]).ravel()).reshape(len(q), len(horizons))
    return base - target, resid, target, g_now


def anchored_error(base_err: np.ndarray, resid: np.ndarray, horizons: np.ndarray, tau_r: float) -> np.ndarray:
    return base_err + resid[:, None] * np.exp(-horizons / tau_r)[None, :]


def fixed_fit_errors(
    p: PData,
    spec: tw.TwinSpec,
    mu: np.ndarray,
    sd: np.ndarray,
    fit_hours: float,
    tau_r: float,
    cfg: dict,
    horizon: float = 60.0,
) -> np.ndarray:
    """60-min forecast errors from hour ``evaluation_from_hour`` on, with the twin fitted once on the first ``fit_hours``."""
    tc = cfg["twin"]
    t_fit, g_fit = p.fit_data(before=fit_hours * 60.0)
    if fit_hours <= 0 or len(t_fit) == 0:
        theta = spec.clip(mu)
    else:
        theta = tw.fit(
            p.inputs.window(-np.inf, fit_hours * 60.0),
            t_fit,
            g_fit,
            spec,
            mu,
            sd,
            tc["sigma_mgdl"],
            max_nfev=tc["max_nfev"],
        )
    idx = sampled_moments(p, tc["forecast_eval_step_min"], tc["evaluation_from_hour"] * 60.0)
    if len(idx) == 0:
        return np.array([])
    h = np.array([horizon])
    base, resid, _, _ = forecast_parts(p, spec, theta, idx, h)
    err = anchored_error(base, resid, h, tau_r)[:, 0]
    return err[~np.isnan(err)]


def nanmean(values) -> float:
    arr = np.asarray(list(values), dtype=float)
    return float(np.nanmean(arr)) if np.any(~np.isnan(arr)) else float("nan")


def rmse(e: np.ndarray) -> float:
    return float(np.sqrt(np.mean(e**2))) if len(e) else float("nan")


# --------------------------------------------------------------------------- events and alerts


def match_alerts(alert_t: np.ndarray, alert_status: np.ndarray, onsets: np.ndarray, cfg: dict) -> dict:
    """Deterministic matching (APPROACH.md §6.2).

    Each alert is matched to the first confirmed onset in (alert, alert + horizon]; an alert is
    matched to at most one event, an event may collect several alerts.
    """
    horizon = cfg["evaluability"]["horizon_min"]
    early = cfg["alerts"]["early_lead_min"]
    onsets = np.sort(np.asarray(onsets, dtype=float))
    alert_t = np.asarray(alert_t, dtype=float)
    j = np.searchsorted(onsets, alert_t, side="right")
    jc = np.minimum(j, max(len(onsets) - 1, 0))
    matched = (j < len(onsets)) & (len(onsets) > 0) & (onsets[jc] <= alert_t + horizon if len(onsets) else False)
    per_event: dict[int, list[float]] = {}
    for i in np.flatnonzero(matched):
        per_event.setdefault(int(j[i]), []).append(alert_t[i])
    event_class, leads = [], []
    for k in range(len(onsets)):
        if k in per_event:
            lead = onsets[k] - min(per_event[k])
            leads.append(lead)
            event_class.append("early" if lead >= early else "late")
        else:
            event_class.append("missed")
    alert_class = np.where(matched, "true", np.where(np.asarray(alert_status) == "unscorable", "unscorable", "false"))
    return {"event_class": event_class, "leads": leads, "alert_class": alert_class.tolist()}


def event_outcome(p: PData, scores: np.ndarray, threshold: float, cfg: dict) -> dict:
    m = p.moments
    t = m["t"].to_numpy(float)
    idx = simulate_alerts(t, scores, m["cov30"].to_numpy(float), threshold, cfg)
    res = match_alerts(t[idx], m["status"].to_numpy()[idx], p.events["onset"].to_numpy(float), cfg)
    ec, ac = res["event_class"], res["alert_class"]
    n = len(ec)
    return {
        "n_events": n,
        "early": ec.count("early"),
        "late": ec.count("late"),
        "missed": ec.count("missed"),
        "leads": res["leads"],
        "alerts": len(idx),
        "true_alerts": ac.count("true"),
        "false_alerts": ac.count("false"),
        "unscorable_alerts": ac.count("unscorable"),
        "days": p.days,
    }


def event_stats(outcomes: dict[str, dict]) -> dict:
    """Per-participant rates (NaN when a participant has no events)."""
    out = {}
    for pid, o in outcomes.items():
        n = o["n_events"]
        out[pid] = {
            "early": o["early"] / n if n else np.nan,
            "late": o["late"] / n if n else np.nan,
            "missed": o["missed"] / n if n else np.nan,
            "false_per_day": o["false_alerts"] / o["days"] if o["days"] > 0 else np.nan,
            "leads": o["leads"],
        }
    return out


def early_rate(outcomes: dict[str, dict]) -> float:
    return nanmean(s["early"] for s in event_stats(outcomes).values())


# --------------------------------------------------------------------------- matrices


def frame(pdatas: dict[str, PData], pids: list[str], tfeats: dict[str, pd.DataFrame] | None) -> pd.DataFrame:
    parts = []
    for pid in pids:
        m = pdatas[pid].moments
        if tfeats is not None and pid in tfeats:
            m = pd.concat([m, tfeats[pid]], axis=1)
        m = m.copy()
        m["pid"] = pid
        parts.append(m)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def scorable(X: pd.DataFrame) -> pd.DataFrame:
    return X[X["status"] != "unscorable"]


# --------------------------------------------------------------------------- inner procedure


@dataclass
class InnerResult:
    spec: tw.TwinSpec
    prior: tw.Prior
    tau_r: float
    band: dict[int, list[float]]
    thresholds: dict[str, float]
    fitted: dict[str, Fitted]
    selected_model: str
    inner_early: dict[str, float]
    extension_info: dict[str, dict]
    dparams: dict[str, np.ndarray] = field(default_factory=dict)
    tfeats: dict[str, pd.DataFrame] = field(default_factory=dict)


def choose_tau_and_band(
    pdatas: dict[str, PData], pids: list[str], spec: tw.TwinSpec, dparams: dict[str, np.ndarray], cfg: dict
) -> tuple[float, dict[int, list[float]]]:
    tc = cfg["twin"]
    horizons = np.array(tc["band_horizons_min"], dtype=float)
    parts = []
    for pid in pids:
        p = pdatas[pid]
        idx = sampled_moments(p, 30.0)
        if len(idx):
            base, resid, _, _ = forecast_parts(p, spec, dparams[pid], idx, horizons)
            parts.append((base, resid))
    if not parts:
        return float(tc["default_tau_r_min"]), {int(h): [-30.0, 30.0] for h in horizons}
    base = np.vstack([b for b, _ in parts])
    resid = np.concatenate([r for _, r in parts])
    h60 = int(np.argmin(np.abs(horizons - 60)))
    best, best_err = None, np.inf
    for tau in tc["tau_r_grid_min"]:
        e = anchored_error(base, resid, horizons, tau)[:, h60]
        e = e[~np.isnan(e)]
        err = rmse(e)
        if err < best_err:
            best, best_err = float(tau), err
    best = best if best is not None else float(tc["default_tau_r_min"])
    e_all = anchored_error(base, resid, horizons, best)
    qlo, qhi = tc["band_quantiles"]
    band = {}
    for k, h in enumerate(horizons):
        col = e_all[:, k]
        col = col[~np.isnan(col)]
        band[int(h)] = [float(np.quantile(col, qlo)), float(np.quantile(col, qhi))] if len(col) else [-30.0, 30.0]
    return best, band


def select_extensions(
    train_pids: list[str], pdatas: dict[str, PData], cfg: dict, cache: dict, seed: int
) -> tuple[tuple[str, ...], dict]:
    """Inner CV: keep an extension only if the 95% CI of (RMSE with - RMSE without) lies below zero."""
    base = tw.TwinSpec(())
    groups = [pdatas[p].group for p in train_pids]
    splits = participant_folds(train_pids, groups, cfg["cv"]["inner_folds"], seed)
    tau = float(cfg["twin"]["default_tau_r_min"])
    fit_h = cfg["twin"]["evaluation_from_hour"]
    keep, info = [], {}
    for ext in cfg["twin"]["extensions"]:
        spec = tw.TwinSpec((ext,))
        diffs: dict[str, float] = {}
        for itr, iva in splits:
            prior_b = prior_for(itr, pdatas, base, cfg, cache)
            prior_e = prior_for(itr, pdatas, spec, cfg, cache)
            for pid in iva:
                p = pdatas[pid]
                eb = fixed_fit_errors(p, base, *prior_b.p1(p.profile), fit_h, tau, cfg)
                ee = fixed_fit_errors(p, spec, *prior_e.p1(p.profile), fit_h, tau, cfg)
                if len(eb) and len(ee):
                    diffs[pid] = rmse(ee) - rmse(eb)
        ids = list(diffs)
        est, lo, hi = boot(ids, lambda s, d=diffs: float(np.mean([d[x] for x in s])), cfg["cv"]["bootstrap"], seed)
        selected = bool(len(ids) > 1 and hi < 0)
        info[ext] = {"rmse_diff": [est, lo, hi], "n": len(ids), "selected": selected}
        if selected:
            keep.append(ext)
    return tuple(keep), info


def inner_procedure(
    train_pids: list[str],
    pdatas: dict[str, PData],
    cfg: dict,
    cache: dict,
    models=MODELS,
    seed: int = 0,
    clock_ok: bool = True,
    select_ext: bool = True,
) -> InnerResult:
    exts, ext_info = select_extensions(train_pids, pdatas, cfg, cache, seed) if select_ext else ((), {})
    spec = tw.TwinSpec(exts)
    prior = prior_for(train_pids, pdatas, spec, cfg, cache)
    need_twin = "M3" in models
    dparams, tfeats = {}, {}
    tau_r = float(cfg["twin"]["default_tau_r_min"])
    band: dict[int, list[float]] = {}
    if need_twin:
        for pid in train_pids:
            dparams[pid] = daily_params(pdatas[pid], spec, *prior.p1(pdatas[pid].profile), cfg)
        tau_r, band = choose_tau_and_band(pdatas, train_pids, spec, dparams, cfg)
        for pid in train_pids:
            tfeats[pid] = twin_features(pdatas[pid], spec, dparams[pid], tau_r)

    groups = [pdatas[p].group for p in train_pids]
    splits = participant_folds(train_pids, groups, cfg["cv"]["inner_folds"], seed + 1)
    oof: dict[str, dict[str, np.ndarray]] = {m: {} for m in models}
    for itr, iva in splits:
        Xtr = scorable(frame(pdatas, itr, tfeats or None))
        for m in models:
            fitted = train(m, Xtr, Xtr["y"].to_numpy(), cfg, clock_ok)
            for pid in iva:
                oof[m][pid] = score(fitted, frame(pdatas, [pid], tfeats or None))
    thresholds, inner_early = {}, {}
    for m in models:
        groups_m = [
            (
                pdatas[pid].moments["t"].to_numpy(float),
                oof[m][pid],
                pdatas[pid].moments["cov30"].to_numpy(float),
                pdatas[pid].days,
            )
            for pid in train_pids
            if pid in oof[m]
        ]
        thresholds[m] = choose_threshold(groups_m, cfg)
        outcomes = {
            pid: event_outcome(pdatas[pid], oof[m][pid], thresholds[m], cfg) for pid in train_pids if pid in oof[m]
        }
        inner_early[m] = early_rate(outcomes)
    ranked = sorted(models, key=lambda m: (-np.nan_to_num(inner_early[m], nan=-1.0), MODELS.index(m)))
    Xall = scorable(frame(pdatas, train_pids, tfeats or None))
    fitted = {m: train(m, Xall, Xall["y"].to_numpy(), cfg, clock_ok) for m in models}
    return InnerResult(spec, prior, tau_r, band, thresholds, fitted, ranked[0], inner_early, ext_info, dparams, tfeats)


# --------------------------------------------------------------------------- outer evaluation


def run_evaluation(
    pdatas: dict[str, PData], cfg: dict, clock_ok: bool, models=MODELS, full: bool = True, log=print
) -> dict:
    seed = cfg["seed"]
    B = cfg["cv"]["bootstrap"]
    pids = sorted(pdatas)
    folds = participant_folds(pids, [pdatas[p].group for p in pids], cfg["cv"]["outer_folds"], seed)
    cache: dict = {}
    tc = cfg["twin"]
    outcomes: dict[str, dict[str, dict]] = {m: {} for m in models}
    rowscores: dict[str, dict[str, np.ndarray]] = {m: {} for m in models}
    fc_err = {
        "twin": {h: {} for h in tc["forecast_horizons_min"]},
        "trend": {h: {} for h in tc["forecast_horizons_min"]},
    }
    fc_pairs = {
        "twin": {h: [] for h in tc["forecast_horizons_min"]},
        "trend": {h: [] for h in tc["forecast_horizons_min"]},
    }
    pers: dict[str, dict[str, dict[int, float]]] = {"P0": {}, "P1": {}}
    pers_mae: dict[str, dict[str, dict[int, float]]] = {"P0": {}, "P1": {}}
    ext_diffs: dict[str, dict[str, float]] = {e: {} for e in tc["extensions"]}
    selection = []
    t_start = time.time()
    for k, (train_pids, test_pids) in enumerate(folds):
        inner = inner_procedure(train_pids, pdatas, cfg, cache, models, seed + 10 * k, clock_ok, select_ext=full)
        selection.append(
            {
                "fold": k + 1,
                "model": inner.selected_model,
                "extensions": list(inner.spec.extensions),
                "thresholds": inner.thresholds,
            }
        )
        tfeats = {}
        dparams = {}
        if "M3" in models:
            for pid in test_pids:
                p = pdatas[pid]
                dparams[pid] = daily_params(p, inner.spec, *inner.prior.p1(p.profile), cfg)
                tfeats[pid] = twin_features(p, inner.spec, dparams[pid], inner.tau_r)
        for pid in test_pids:
            p = pdatas[pid]
            X = frame(pdatas, [pid], tfeats or None)
            for m in models:
                s = score(inner.fitted[m], X)
                rowscores[m][pid] = s
                outcomes[m][pid] = event_outcome(p, s, inner.thresholds[m], cfg)
        if not full:
            log(f"  fold {k + 1}/{len(folds)} done ({time.time() - t_start:.0f}s)")
            continue
        horizons = np.array(tc["forecast_horizons_min"], dtype=float)
        for pid in test_pids:
            p = pdatas[pid]
            idx = sampled_moments(p, tc["forecast_eval_step_min"])
            if len(idx) == 0:
                continue
            if "M3" in models:
                base, resid, target, g_now = forecast_parts(p, inner.spec, dparams[pid], idx, horizons)
                twin_e = anchored_error(base, resid, horizons, inner.tau_r)
            else:
                twin_e = np.full((len(idx), len(horizons)), np.nan)
                target = p.targets((p.moments["t"].to_numpy(float)[idx][:, None] + horizons[None, :]).ravel()).reshape(
                    len(idx), len(horizons)
                )
                g_now = p.moments["g_now"].to_numpy(float)[idx]
            slope = p.moments["slope_30"].fillna(0.0).to_numpy(float)[idx]
            trend_pred = g_now[:, None] + horizons[None, :] * slope[:, None]
            for j, h in enumerate(tc["forecast_horizons_min"]):
                ok = ~np.isnan(target[:, j])
                for name, err, pred in (
                    ("twin", twin_e[:, j], target[:, j] + twin_e[:, j]),
                    ("trend", trend_pred[:, j] - target[:, j], trend_pred[:, j]),
                ):
                    good = ok & ~np.isnan(err)
                    fc_err[name][h][pid] = err[good]
                    fc_pairs[name][h].extend(zip(target[good, j], pred[good], strict=True))
        base_spec = tw.TwinSpec(())
        prior_base = prior_for(train_pids, pdatas, base_spec, cfg, cache)
        priors_ext = {e: prior_for(train_pids, pdatas, tw.TwinSpec((e,)), cfg, cache) for e in tc["extensions"]}
        for pid in test_pids:
            p = pdatas[pid]
            for label, (mu, sd) in (("P0", inner.prior.p0()), ("P1", inner.prior.p1(p.profile))):
                pers[label].setdefault(pid, {})
                pers_mae[label].setdefault(pid, {})
                for hrs in tc["personalisation_hours"]:
                    e = fixed_fit_errors(p, inner.spec, mu, sd, hrs, inner.tau_r, cfg)
                    pers[label][pid][hrs] = rmse(e)
                    pers_mae[label][pid][hrs] = float(np.mean(np.abs(e))) if len(e) else float("nan")
            eb = fixed_fit_errors(p, base_spec, *prior_base.p1(p.profile), tc["evaluation_from_hour"], inner.tau_r, cfg)
            for e_name, pr in priors_ext.items():
                ee = fixed_fit_errors(
                    p, tw.TwinSpec((e_name,)), *pr.p1(p.profile), tc["evaluation_from_hour"], inner.tau_r, cfg
                )
                if len(eb) and len(ee):
                    ext_diffs[e_name][pid] = rmse(ee) - rmse(eb)
        log(
            f"  fold {k + 1}/{len(folds)} done ({time.time() - t_start:.0f}s); inner selected {inner.selected_model}, extensions {list(inner.spec.extensions) or 'none'}"
        )

    results: dict = {"models": {}, "selection": selection}
    for m in models:
        stats = event_stats(outcomes[m])
        ids = list(stats)
        ev_ids = [pid for pid in ids if outcomes[m][pid]["n_events"] > 0]

        def mean_of(key, st=stats):
            return lambda s: nanmean(st[x][key] for x in s)

        def pooled_leads(s, st=stats):
            leads = [lead for x in s for lead in st[x]["leads"]]
            return float(np.median(leads)) if leads else float("nan")

        def rowmetric(fn, mm=m):
            def stat(s):
                y = np.concatenate([pdatas[x].moments["y"].to_numpy(float) for x in s])
                sc = np.concatenate([rowscores[mm][x] for x in s])
                ok = ~np.isnan(y)
                y, sc = y[ok], sc[ok]
                if len(np.unique(y)) < 2:
                    return float("nan")
                return float(fn(y, sc))

            return stat

        res = {
            "early_rate": boot(ev_ids, mean_of("early"), B, seed),
            "late_rate": boot(ev_ids, mean_of("late"), B, seed),
            "miss_rate": boot(ev_ids, mean_of("missed"), B, seed),
            "median_lead_min": boot(ev_ids, pooled_leads, B, seed),
            "false_alerts_per_day": boot(ids, mean_of("false_per_day"), B, seed),
            "auprc": boot(ids, rowmetric(average_precision_score), B, seed),
            "auroc": boot(ids, rowmetric(roc_auc_score), B, seed),
            "events": int(sum(o["n_events"] for o in outcomes[m].values())),
            "alerts": int(sum(o["alerts"] for o in outcomes[m].values())),
            "unscorable_alerts": int(sum(o["unscorable_alerts"] for o in outcomes[m].values())),
            "participants_with_events": len(ev_ids),
        }
        if m != "M0":
            res["brier_uncalibrated"] = boot(
                ids, rowmetric(lambda y, s: brier_score_loss(y, np.clip(s, 0, 1))), B, seed
            )[0]
        by_group = {}
        for grp in sorted({pdatas[x].group for x in ids}):
            gids = [x for x in ids if pdatas[x].group == grp]
            by_group[grp] = {
                "early_rate": nanmean(stats[x]["early"] for x in gids),
                "false_alerts_per_day": nanmean(stats[x]["false_per_day"] for x in gids),
                "participants": len(gids),
            }
        res["by_group"] = by_group
        results["models"][m] = res
    if full:
        results["extensions"] = {}
        for e_name, diffs in ext_diffs.items():
            ids = list(diffs)
            results["extensions"][e_name] = {
                "selected_in_folds": sum(e_name in s["extensions"] for s in selection),
                "rmse_diff_mgdl": boot(ids, lambda s, d=diffs: float(np.mean([d[x] for x in s])), B, seed),
                "participants": len(ids),
            }
        results["personalisation"] = {}
        for label in ("P0", "P1"):
            results["personalisation"][label] = {}
            for hrs in tc["personalisation_hours"]:
                ids = [x for x in pers[label] if not np.isnan(pers[label][x][hrs])]
                results["personalisation"][label][str(hrs)] = {
                    "rmse": boot(
                        ids, lambda s, lb=label, hh=hrs: float(np.mean([pers[lb][x][hh] for x in s])), B, seed
                    ),
                    "mae": float(np.mean([pers_mae[label][x][hrs] for x in ids])) if ids else float("nan"),
                }
        results["forecast"] = {}
        for name in ("twin", "trend"):
            results["forecast"][name] = {}
            for h in tc["forecast_horizons_min"]:
                errs = fc_err[name][h]
                ids = [x for x in errs if len(errs[x])]
                pairs = fc_pairs[name][h]
                zones = [clarke_zone(r, q) for r, q in pairs]
                results["forecast"][name][str(h)] = {
                    "rmse": boot(ids, lambda s, e=errs: rmse(np.concatenate([e[x] for x in s])), B, seed),
                    "mae": boot(
                        ids, lambda s, e=errs: float(np.mean(np.abs(np.concatenate([e[x] for x in s])))), B, seed
                    ),
                    "clarke_ab_pct": 100.0 * float(np.mean([z in ("A", "B") for z in zones]))
                    if zones
                    else float("nan"),
                    "points": len(pairs),
                }
    return results


# --------------------------------------------------------------------------- descriptive


def describe(td, pdatas: dict[str, PData], cfg: dict) -> dict:
    prof = td.profiles
    n_events = [len(p.events) for p in pdatas.values()]
    moments = (
        pd.concat([p.moments[["status"]] for p in pdatas.values()]) if pdatas else pd.DataFrame(columns=["status"])
    )
    tl_rows = sum(len(p.tl) for p in pdatas.values())
    imputed = sum(int((p.tl["src"] == "imputed").sum()) for p in pdatas.values())
    primary = cfg["sensors"]["primary"]
    cens = td.cgm[(td.cgm["sensor"] == primary) & td.cgm["pid"].isin(list(pdatas))]["censored"]
    excluded = prof[~prof.get("included", pd.Series(True, index=prof.index)).astype(bool)]
    return {
        "participants_total": int(len(prof)),
        "participants_in_analysis": len(pdatas),
        "participants_excluded": {pid: str(r.get("exclusion_reason", "")) for pid, r in excluded.iterrows()},
        "groups": pd.Series({pid: p.group for pid, p in pdatas.items()}).value_counts().to_dict(),
        "days_of_cgm": float(round(sum(p.days for p in pdatas.values()), 1)),
        "primary_coverage_mean": float(prof.loc[list(pdatas), "primary_coverage"].mean())
        if "primary_coverage" in prof
        else float("nan"),
        "imputed_share": float(imputed / tl_rows) if tl_rows else float("nan"),
        "censored_low": int((cens == -1).sum()),
        "censored_high": int((cens == 1).sum()),
        "eligible_moments": int(len(moments)),
        "evaluable_windows": int((moments["status"] != "unscorable").sum()),
        "unscorable_windows": int((moments["status"] == "unscorable").sum()),
        "events_total": int(sum(n_events)),
        "events_per_participant": {
            "median": float(np.median(n_events)) if n_events else float("nan"),
            "min": int(min(n_events, default=0)),
            "max": int(max(n_events, default=0)),
        },
    }


def propensity(pdatas: dict[str, PData], cfg: dict) -> dict:
    """Profile-only analysis: HbA1c vs each participant's excursion rate (baseline propensity)."""
    ids = [pid for pid, p in pdatas.items() if p.days > 0]
    if len(ids) < 4:
        return {"spearman_hba1c_vs_events_per_day": [float("nan")] * 3, "participants": len(ids)}
    rate = {pid: len(pdatas[pid].events) / pdatas[pid].days for pid in ids}
    hb = {pid: float(pdatas[pid].profile.get("hba1c_pct", np.nan)) for pid in ids}

    def stat(s):
        x = [hb[i] for i in s]
        y = [rate[i] for i in s]
        if len(set(x)) < 2 or len(set(y)) < 2:
            return float("nan")
        return float(spearmanr(x, y).statistic)

    return {
        "spearman_hba1c_vs_events_per_day": boot(ids, stat, cfg["cv"]["bootstrap"], cfg["seed"]),
        "participants": len(ids),
    }


def identifiability(pdatas: dict[str, PData], cfg: dict, max_participants: int = 30) -> dict:
    """Spread of the base twin's parameters when refitted on resampled days (descriptive)."""
    spec = tw.TwinSpec(())
    mu, sd = spec.generic_prior()
    rng = np.random.default_rng(cfg["seed"])
    B = cfg["twin"]["identifiability_bootstraps"]
    sigma, nfev = cfg["twin"]["sigma_mgdl"], cfg["twin"]["max_nfev"]
    spreads = {"baseline_mgdl": [], "rise_per_g_carb": [], "time_to_peak_min": []}
    for pid in sorted(pdatas)[:max_participants]:
        p = pdatas[pid]
        t_all, g_all = p.fit_data()
        if len(t_all) == 0:
            continue
        full = tw.fit(p.inputs, t_all, g_all, spec, mu, sd, sigma, max_nfev=2 * nfev)
        day = np.floor(t_all / 1440.0).astype(int)
        days = np.unique(day)
        fits = []
        for _ in range(B):
            pick = rng.choice(days, len(days), replace=True)
            sel = np.concatenate([np.flatnonzero(day == d) for d in pick])
            theta = tw.fit(p.inputs, t_all[sel], g_all[sel], spec, mu, sd, sigma, x0=full, max_nfev=nfev)
            fits.append(spec.readable(theta))
        df = pd.DataFrame(fits)
        for key in spreads:
            spreads[key].append(float(df[key].std(ddof=1)))
    return {
        k: {"median_sd": float(np.median(v)) if v else float("nan"), "participants": len(v)} for k, v in spreads.items()
    }
