"""Evidence track: load CGMacros (PhysioNet, CC BY-NC-SA 4.0) into the canonical tables.

The dataset is never redistributed. ``scripts/download_cgmacros.sh`` puts it under
``data/raw/cgmacros``; this loader searches that folder recursively for ``bio.csv`` and the
per-participant ``CGMacros-0XX.csv`` files.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .align import estimate_sleep, wear_coverage
from .schema import (
    ACTIVITY_COLUMNS,
    CGM_COLUMNS,
    MEAL_COLUMNS,
    SLEEP_COLUMNS,
    TrackData,
    ValidationError,
)
from .validate import (
    censor_flags,
    hba1c_groups,
    meal_order_fraction,
    reconstruct_native,
    require_increasing,
    scale_mets,
    sensor_agreement,
)


def _norm(name: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _find(columns: dict[str, str], exact=(), startswith=(), contains_all=()) -> str | None:
    for key, original in columns.items():
        if key in exact:
            return original
    for key, original in columns.items():
        if any(key.startswith(s) for s in startswith):
            return original
    for key, original in columns.items():
        if contains_all and all(part in key for part in contains_all):
            return original
    return None


def _pid_from(text: object) -> str:
    digits = re.findall(r"\d+", str(text))
    if not digits:
        raise ValidationError(f"Cannot read a participant number from {text!r}.")
    return f"CGM-{int(digits[-1]):03d}"


def find_files(raw_dir: Path) -> tuple[Path, dict[str, Path]]:
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise ValidationError(
            f"CGMacros not found at {raw_dir}. Run `bash scripts/download_cgmacros.sh` first (see data/README.md)."
        )
    bios = sorted(raw_dir.rglob("bio.csv"))
    if len(bios) != 1:
        raise ValidationError(f"Expected exactly one bio.csv under {raw_dir}, found {len(bios)}.")
    files: dict[str, Path] = {}
    for path in sorted(raw_dir.rglob("*.csv")):
        if re.fullmatch(r"(?i)cgmacros-?\d+\.csv", path.name):
            pid = _pid_from(path.stem)
            if pid in files:
                raise ValidationError(f"Two files for {pid}: {files[pid]} and {path}.")
            files[pid] = path
    if not files:
        raise ValidationError(f"No CGMacros-0XX.csv participant files under {raw_dir}.")
    return bios[0], files


def load_bio(path: Path, cfg: dict) -> tuple[pd.DataFrame, list[tuple[str, str, str]]]:
    bio = pd.read_csv(path)
    cols = {_norm(c): c for c in bio.columns}
    checks: list[tuple[str, str, str]] = []
    id_col = _find(cols, exact=("subject", "subjectid", "participant", "participantid", "id", "pid"))
    if id_col is None:
        raise ValidationError("bio.csv has no participant ID column, so labs cannot be joined.")
    mapping = {
        "age": _find(cols, exact=("age",)),
        "sex": _find(cols, exact=("gender", "sex")),
        "bmi": _find(cols, exact=("bmi",)),
        "weight": _find(cols, exact=("bodyweight", "weight")),
        "height": _find(cols, exact=("height",)),
        "hba1c_raw": _find(cols, contains_all=("a1c",)),
        "fasting_glucose": _find(cols, contains_all=("fasting", "glu")),
        "fasting_insulin": _find(cols, exact=("insulin", "fastinginsulin")),
        "triglycerides": _find(cols, contains_all=("triglyc",)),
        "total_chol": _find(cols, exact=("cholesterol", "totalcholesterol")),
        "hdl": _find(cols, exact=("hdl",)),
        "ldl": _find(cols, startswith=("ldl",)),
        "vldl": _find(cols, startswith=("vldl",)),
    }
    for required in ("hba1c_raw", "fasting_glucose"):
        if mapping[required] is None:
            raise ValidationError(f"bio.csv has no column for {required}.")
    prof = pd.DataFrame({"pid": bio[id_col].map(_pid_from)})
    for key, col in mapping.items():
        prof[key] = bio[col] if col is not None else np.nan
    if prof["pid"].duplicated().any():
        raise ValidationError("bio.csv lists a participant more than once.")
    for key in (
        "age",
        "bmi",
        "weight",
        "height",
        "hba1c_raw",
        "fasting_glucose",
        "fasting_insulin",
        "triglycerides",
        "total_chol",
        "hdl",
        "ldl",
        "vldl",
    ):
        prof[key] = pd.to_numeric(prof[key], errors="coerce")
    codes = cfg["cgmacros"]["lab_error_codes"]
    for lab, code in codes.items():
        n = int((prof[lab] == code).sum())
        prof.loc[prof[lab] == code, lab] = np.nan
        checks.append((f"Lab error code {lab.upper()}={code}", "pass", f"{n} values set to missing"))
    prof["sex"] = prof["sex"].astype(str).str.upper().str[0].replace({"N": np.nan})
    return prof.set_index("pid"), checks


def _participant(path: Path, pid: str, cfg: dict) -> dict:
    df = pd.read_csv(path, low_memory=False)
    cols = {_norm(c): c for c in df.columns}
    ts_col = _find(cols, exact=("timestamp", "time", "datetime"))
    if ts_col is None:
        raise ValidationError(f"{path.name}: no Timestamp column.")
    times = pd.to_datetime(df[ts_col], errors="coerce")
    require_increasing(times, path.name)
    sens = cfg["sensors"]
    tol = cfg["cgmacros"]["interpolation_tolerance_mgdl"]
    out: dict = {"modes": {}}
    cgm_parts = []
    for sensor, keys in (("dexcom", ("dexcomgl", "dexcom")), ("libre", ("libregl", "libre"))):
        col = _find(cols, exact=keys, startswith=keys)
        if col is None:
            out["modes"][sensor] = "absent"
            continue
        interval = sens[f"{sensor}_interval_min"]
        mask, mode = reconstruct_native(times, df[col], interval, tol)
        out["modes"][sensor] = mode
        if mode == "unknown":
            raise ValidationError(
                f"{path.name}: native {sensor} readings could not be identified from the "
                "1-minute rows (not blank, carried forward or linearly interpolated)."
            )
        g = pd.to_numeric(df[col], errors="coerce").to_numpy(float)[mask]
        part = pd.DataFrame({"pid": pid, "time": times[mask].to_numpy(), "glucose": g, "sensor": sensor})
        part["censored"] = censor_flags(g, sens["glucose_low_limit"], sens["glucose_high_limit"])
        cgm_parts.append(part)
    if not cgm_parts:
        raise ValidationError(f"{path.name}: no Dexcom or Libre column.")
    out["cgm"] = pd.concat(cgm_parts, ignore_index=True)[CGM_COLUMNS]

    hr_col = _find(cols, exact=("hr", "heartrate"))
    mets_col = _find(cols, exact=("mets", "met"))
    act = pd.DataFrame({"pid": pid, "time": times})
    act["hr"] = pd.to_numeric(df[hr_col], errors="coerce") if hr_col else np.nan
    if mets_col:
        act["mets"] = scale_mets(df[mets_col], cfg["cgmacros"]["mets_scale"], tuple(cfg["cgmacros"]["mets_plausible"]))
    else:
        act["mets"] = np.nan
    act["dur_min"] = 1.0
    out["activity"] = act.dropna(subset=["hr", "mets"], how="all")[ACTIVITY_COLUMNS]

    carbs_col = _find(cols, exact=("carbs", "carbohydrates", "carbohydrate"))
    if carbs_col is None:
        raise ValidationError(f"{path.name}: no Carbs column.")
    meal_cols = {
        "meal_type": _find(cols, exact=("mealtype",)),
        "calories": _find(cols, exact=("calories", "mealcalories")),
        "carbs": carbs_col,
        "protein": _find(cols, exact=("protein",)),
        "fat": _find(cols, exact=("fat",)),
        "fiber": _find(cols, exact=("fiber", "fibre")),
        "fraction": _find(cols, contains_all=("consumed",)),
    }
    rows = df[carbs_col].notna()
    if meal_cols["meal_type"]:
        rows |= df[meal_cols["meal_type"]].notna()
    meals = pd.DataFrame({"pid": pid, "time": times[rows].to_numpy()})
    for key, col in meal_cols.items():
        meals[key] = df.loc[rows, col].to_numpy() if col else np.nan
    for key in ("calories", "carbs", "protein", "fat", "fiber", "fraction"):
        meals[key] = pd.to_numeric(meals[key], errors="coerce")
    out["fraction_missing"] = int(meals["fraction"].isna().sum())
    frac = meals["fraction"]
    if frac.notna().any() and frac.max() > 1.5:
        frac = frac / 100.0
    meals["fraction"] = frac.fillna(1.0).clip(0, 1)
    # A meal may appear on two rows (start and end photos); keep the first of a repeat.
    meals = meals.sort_values("time")
    window = pd.Timedelta(minutes=cfg["cgmacros"]["meal_dedup_window_min"])
    keep, last = [], None
    for _, r in meals.iterrows():
        sig = (r["meal_type"], r["carbs"], r["protein"], r["fat"])
        if last is not None and sig == last[0] and r["time"] - last[1] <= window:
            keep.append(False)
            continue
        keep.append(True)
        last = (sig, r["time"])
    out["meal_duplicates"] = int(len(keep) - sum(keep))
    meals = meals[np.array(keep, dtype=bool)] if len(keep) else meals
    meals = meals[meals["carbs"].notna()].copy()
    for macro in ("carbs", "protein", "fat", "fiber"):
        meals[f"{macro}_eaten"] = meals[macro].fillna(0) * meals["fraction"]
    out["meals"] = meals[MEAL_COLUMNS]
    return out


def ingest(raw_dir: Path, cfg: dict) -> TrackData:
    """Load, validate and tag the evidence track. Raises ValidationError on a stop rule."""
    bio_path, files = find_files(raw_dir)
    prof, lab_checks = load_bio(bio_path, cfg)
    td = TrackData(
        "evidence",
        prof,
        pd.DataFrame(columns=CGM_COLUMNS),
        pd.DataFrame(columns=ACTIVITY_COLUMNS),
        pd.DataFrame(columns=MEAL_COLUMNS),
        pd.DataFrame(columns=SLEEP_COLUMNS),
    )
    missing_files = sorted(set(prof.index) - set(files))
    extra_files = sorted(set(files) - set(prof.index))
    if missing_files or extra_files:
        raise ValidationError(
            f"Join check failed: bio.csv rows without a participant file {missing_files}; "
            f"participant files without a bio.csv row {extra_files}."
        )
    td.add_check("Join: bio.csv rows to participant files", "pass", f"{len(files)} participants matched one-to-one")

    groups, detail = hba1c_groups(
        prof["hba1c_raw"],
        tuple(cfg["cgmacros"]["hba1c_thresholds_pct"]),
        cfg["cgmacros"].get("expected_group_counts"),
    )
    td.add_check("HbA1c units (stop rule)", "pass", detail)
    prof["hba1c_pct"] = prof.pop("hba1c_raw")
    prof["group"] = groups
    for name, status, det in lab_checks:
        td.add_check(name, status, det)

    cgm, act, meals, sleeps, modes = [], [], [], [], {}
    dup_total = frac_missing = 0
    for pid in sorted(files):
        part = _participant(files[pid], pid, cfg)
        cgm.append(part["cgm"])
        act.append(part["activity"])
        meals.append(part["meals"])
        modes[pid] = part["modes"]
        dup_total += part["meal_duplicates"]
        frac_missing += part["fraction_missing"]
        sleeps.append(estimate_sleep(part["activity"], cfg))
    td.cgm = pd.concat(cgm, ignore_index=True)
    td.activity = pd.concat(act, ignore_index=True)
    td.meals = pd.concat(meals, ignore_index=True)
    td.sleep = pd.concat(sleeps, ignore_index=True) if sleeps else td.sleep

    mode_counts = pd.DataFrame(modes).T
    for sensor in ("dexcom", "libre"):
        if sensor in mode_counts:
            vc = mode_counts[sensor].value_counts().to_dict()
            td.add_check(f"Native sampling: {sensor}", "pass", f"1-minute rows between readings were {vc}")
    td.add_check("Meals: repeated rows", "info", f"{dup_total} duplicate meal rows (start/end photos) merged")
    td.add_check("Meals: proportion eaten", "info", f"{frac_missing} meals without a recorded proportion; assumed 100%")

    # Primary sensor coverage, inclusion and sensor agreement.
    primary = cfg["sensors"]["primary"]
    interval = cfg["sensors"][f"{primary}_interval_min"]
    cov, mard = {}, {}
    for pid in prof.index:
        pc = td.cgm[(td.cgm["pid"] == pid) & (td.cgm["sensor"] == primary)]
        oc = td.cgm[(td.cgm["pid"] == pid) & (td.cgm["sensor"] != primary)]
        cov[pid] = wear_coverage(pc["time"], interval)
        mard[pid] = sensor_agreement(pc, oc)
    prof["primary_coverage"] = pd.Series(cov)
    prof["sensor_mard_pct"] = pd.Series(mard)
    prof["included"] = prof["primary_coverage"] >= cfg["sensors"]["primary_min_coverage"]
    prof["exclusion_reason"] = np.where(prof["included"], "", f"{primary} coverage below threshold")
    td.add_check(
        f"Primary sensor ({primary}) coverage",
        "pass" if prof["included"].all() else "warn",
        f"{int(prof['included'].sum())} of {len(prof)} participants have >= "
        f"{cfg['sensors']['primary_min_coverage']:.0%} of expected readings; the rest are excluded "
        "from the primary analysis.",
    )
    censored = td.cgm[td.cgm["sensor"] == primary]["censored"]
    td.add_check(
        "Censored glucose readings",
        "info",
        f"{int((censored == -1).sum())} at the floor (<= 40), {int((censored == 1).sum())} at the ceiling (>= 400)",
    )
    frac, ndays = meal_order_fraction(td.meals)
    clock_ok = bool(ndays > 0 and frac >= cfg["cgmacros"]["meal_order_min_fraction"])
    td.add_check(
        "Clock check (meal order)",
        "pass" if clock_ok else "warn",
        f"{frac:.0%} of {ndays} days log breakfast < lunch < dinner; time-of-day features "
        + ("used." if clock_ok else "dropped."),
    )

    prof["homa_ir"] = prof["fasting_insulin"] * prof["fasting_glucose"] / 405.0
    prof["tg_hdl"] = prof["triglycerides"] / prof["hdl"]
    td.profiles = prof
    td.meta.update({"clock_ok": clock_ok, "sampling_modes": modes, "track": "evidence"})
    return td
