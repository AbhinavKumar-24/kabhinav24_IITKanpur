"""Orchestration: build a track, evaluate it, write reports, train the dashboard bundle.

The evidence and synthetic tracks always run as separate executions with separate outputs.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import __version__
from .config import repo_path
from .evaluate import describe, identifiability, inner_procedure, propensity, run_evaluation
from .features import prepare_track
from .ingest_cgmacros import ingest
from .models import MODELS, Fitted
from .report import data_summary, results_md, write_json
from .schema import Check, TrackData
from .simulate import simulate_track


def load_synthetic_profiles(cfg: dict) -> pd.DataFrame:
    path = repo_path(cfg, cfg["paths"]["synthetic_dir"], "synthetic_profiles.csv")
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `bash scripts/generate_synthea.sh` to create it.")
    return pd.read_csv(path)


def stratified_first(profiles: pd.DataFrame, n: int) -> list[str]:
    """Deterministic subset keeping the group mix (first participants of each group)."""
    shares = profiles["group"].value_counts(normalize=True)
    picks = []
    for group, share in shares.items():
        k = max(1, int(round(n * share)))
        picks += profiles[profiles["group"] == group]["pid"].head(k).tolist()
    return picks[:n] if len(picks) > n else picks


def synthetic_trackdata(cfg: dict, smoke: bool = False) -> TrackData:
    prof = load_synthetic_profiles(cfg)
    syn = cfg["synthetic"]
    days = syn["days"]
    if smoke:
        prof = prof[prof["pid"].isin(stratified_first(prof, syn["smoke"]["n_patients"]))]
        days = syn["smoke"]["days"]
    return simulate_track(prof, cfg, days=days)


def output_dirs(cfg: dict, track: str, out: str | None) -> dict[str, Path]:
    if out:
        base = Path(out)
        return {
            "reports": base / "reports" / track,
            "models": base / "models" / track,
            "processed": base / "processed" / track,
        }
    reports = repo_path(cfg, cfg["paths"]["reports"])
    return {
        "reports": reports if track == "evidence" else reports / track,
        "models": repo_path(cfg, cfg["paths"]["models"], track),
        "processed": repo_path(cfg, cfg["paths"]["processed"], track),
    }


# ----------------------------------------------------------------------------- persistence


def save_trackdata(td: TrackData, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    td.profiles.reset_index(names="pid").to_parquet(folder / "profiles.parquet", index=False)
    for name in ("cgm", "activity", "meals", "sleep"):
        getattr(td, name).to_parquet(folder / f"{name}.parquet", index=False)
    meta = {k: v for k, v in td.meta.items() if k != "truth"}
    payload = {"track": td.track, "meta": meta, "checks": [c.__dict__ for c in td.checks]}
    (folder / "meta.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def load_trackdata(folder: Path) -> TrackData:
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    tables = {name: pd.read_parquet(folder / f"{name}.parquet") for name in ("cgm", "activity", "meals", "sleep")}
    prof = pd.read_parquet(folder / "profiles.parquet").set_index("pid")
    td = TrackData(meta["track"], prof, tables["cgm"], tables["activity"], tables["meals"], tables["sleep"])
    td.checks = [Check(**c) for c in meta["checks"]]
    td.meta = meta["meta"]
    return td


def save_bundle(bundle: dict, fitted: dict[str, Fitted], folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for m, f in fitted.items():
        path = folder / f"model_{m}.txt"
        if f.booster is not None:
            f.booster.save_model(str(path))
        elif path.exists():
            path.unlink()
    write_json(folder / "bundle.json", bundle)


def load_bundle(folder: Path) -> tuple[dict, dict[str, Fitted]]:
    bundle = json.loads((Path(folder) / "bundle.json").read_text(encoding="utf-8"))
    fitted = {}
    for m in MODELS:
        info = bundle["models"][m]
        path = Path(folder) / f"model_{m}.txt"
        booster = lgb.Booster(model_file=str(path)) if path.exists() else None
        fitted[m] = Fitted(m, info["features"], booster, info.get("constant"))
    return bundle, fitted


def build_bundle(td: TrackData, pdatas: dict, cfg: dict, clock_ok: bool) -> tuple[dict, dict[str, Fitted]]:
    """The dashboard model: the same inner procedure, applied to all participants."""
    pids = sorted(pdatas)
    inner = inner_procedure(pids, pdatas, cfg, {}, MODELS, cfg["seed"] + 999, clock_ok)
    bundle = {
        "track": td.track,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "glucotwin_version": __version__,
        "participants": pids,
        "selected_model": inner.selected_model,
        "inner_early_warning_rate": inner.inner_early,
        "thresholds": inner.thresholds,
        "clock_ok": clock_ok,
        "spec": list(inner.spec.extensions),
        "extension_selection": inner.extension_info,
        "prior": inner.prior.to_dict(),
        "tau_r": inner.tau_r,
        "band": {str(k): v for k, v in inner.band.items()},
        "daily_params": {pid: np.asarray(v).tolist() for pid, v in inner.dparams.items()},
        "models": {m: {"features": f.features, "constant": f.constant} for m, f in inner.fitted.items()},
    }
    return bundle, inner.fitted


# ----------------------------------------------------------------------------- run


SENSITIVITY_VARIANTS = {
    "without imputed features": {"use_imputed": False},
    "Libre as primary sensor (15-min)": {"sensor": "libre"},
    "5-minute event persistence": {"persistence": 5.0},
    "windows with censored readings excluded": {"exclude_censored": True},
}


def run_sensitivity(td: TrackData, cfg: dict, pids: list[str], clock_ok: bool, log=print) -> dict:
    out = {}
    for name, variant in SENSITIVITY_VARIANTS.items():
        if variant.get("sensor") == "libre" and not (td.cgm["sensor"] == "libre").any():
            continue
        log(f"Sensitivity: {name}")
        pdv = prepare_track(td, cfg, pids, **variant)
        res = run_evaluation(pdv, cfg, clock_ok, models=("M0", "M1", "M2"), full=False, log=log)
        out[name] = {"participants": len(pdv), "models": res["models"]}
    return out


def run(
    track: str, cfg: dict, out: str | None = None, smoke: bool = False, sensitivity: bool = False, log=print
) -> dict:
    t0 = time.time()
    dirs = output_dirs(cfg, track, out)
    if track == "evidence":
        log("Loading and validating CGMacros ...")
        td = ingest(repo_path(cfg, cfg["paths"]["raw_cgmacros"]), cfg)
    else:
        log("Building the synthetic track (Synthea profiles + simulated streams) ...")
        td = synthetic_trackdata(cfg, smoke)
    clock_ok = bool(td.meta.get("clock_ok", True))
    pdatas = prepare_track(td, cfg)
    eval_pids = sorted(pdatas)
    if track == "synthetic":
        n_eval = cfg["synthetic"]["smoke"]["eval_patients"] if smoke else cfg["synthetic"]["eval_patients"]
        prof = td.profiles.reset_index(names="pid")
        eval_pids = [p for p in stratified_first(prof[prof["pid"].isin(eval_pids)], n_eval) if p in pdatas]
    ev = {pid: pdatas[pid] for pid in eval_pids}
    log(f"Prepared {len(pdatas)} participants; evaluating {len(ev)} ({time.time() - t0:.0f}s)")
    desc = describe(td.subset(eval_pids), ev, cfg)

    results = run_evaluation(ev, cfg, clock_ok, log=log)
    results["propensity"] = propensity(ev, cfg)
    results["identifiability"] = identifiability(ev, cfg)
    if sensitivity:
        results["sensitivity"] = run_sensitivity(td, cfg, eval_pids, clock_ok, log)
    dirs["reports"].mkdir(parents=True, exist_ok=True)
    (dirs["reports"] / "data_summary.md").write_text(data_summary(track, td.checks, desc), encoding="utf-8")
    (dirs["reports"] / "results.md").write_text(results_md(track, results, desc, cfg), encoding="utf-8")
    write_json(
        dirs["reports"] / "metrics.json",
        {"track": track, "counts": desc, "results": results, "glucotwin_version": __version__},
    )
    log(f"Reports written to {dirs['reports']} ({time.time() - t0:.0f}s)")

    log("Training the dashboard bundle ...")
    bundle, fitted = build_bundle(td, ev, cfg, clock_ok)
    save_bundle(bundle, fitted, dirs["models"])
    save_trackdata(td, dirs["processed"])
    log(f"Bundle saved to {dirs['models']}; dashboard model {bundle['selected_model']} ({time.time() - t0:.0f}s)")
    return {"results": results, "counts": desc, "bundle": bundle, "dirs": {k: str(v) for k, v in dirs.items()}}
