"""Synthetic streams.

Two uses, never mixed with real data:

* ``simulate_track`` builds the synthetic track: Synthea profiles + simulated CGM, heart rate,
  activity, sleep and HRV generated from Indian meal patterns. Demo only, never evidence: its
  glucose comes from a meal-response model like the twin itself.
* ``write_cgmacros_like`` writes small files shaped like CGMacros (bio.csv and 1-minute
  participant CSVs) so tests and CI can exercise the evidence loader without the real dataset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from .schema import SLEEP_COLUMNS, TrackData
from .validate import censor_flags

SLOT_HOURS = {"breakfast": (7.5, 9.5), "lunch": (12.5, 14.5), "dinner": (19.5, 21.5), "snack": (16.5, 17.5)}

# CGMacros-like meals for fixtures only (shake breakfasts, restaurant lunches, own dinners).
FIXTURE_MEALS = [
    {"name": "shake low carb", "slot": "breakfast", "carbs": 25, "protein": 25, "fat": 10, "fiber": 3, "kcal": 290},
    {"name": "shake high carb", "slot": "breakfast", "carbs": 75, "protein": 20, "fat": 8, "fiber": 4, "kcal": 450},
    {"name": "restaurant bowl", "slot": "lunch", "carbs": 95, "protein": 35, "fat": 25, "fiber": 12, "kcal": 740},
    {"name": "restaurant salad", "slot": "lunch", "carbs": 40, "protein": 30, "fat": 22, "fiber": 10, "kcal": 480},
    {"name": "home dinner", "slot": "dinner", "carbs": 70, "protein": 30, "fat": 25, "fiber": 6, "kcal": 640},
    {"name": "pasta dinner", "slot": "dinner", "carbs": 100, "protein": 25, "fat": 20, "fiber": 5, "kcal": 700},
]


def sim_params(profile: pd.Series, sim: dict, rng: np.random.Generator) -> dict:
    """Simulation truth for one synthetic person (rules chosen by us, documented in config.yaml)."""
    hb = float(profile["hba1c_pct"])
    return {
        "b0": float(profile["fasting_glucose"]) * rng.uniform(0.95, 1.02),
        "kc": (sim["kc_base"] + sim["kc_per_hba1c"] * max(hb - 5.0, 0.0))
        * float(np.exp(rng.normal(0, sim["kc_noise_sd"]))),
        "tau": float(
            np.clip(sim["tau_base_min"] + sim["tau_per_hba1c"] * (hb - 5.0), 35, 90) * np.exp(rng.normal(0, 0.1))
        ),
        "beta": sim["beta"],
        "ka": sim["ka"],
        "ks": sim["ks"],
    }


def _native_series(true_g: np.ndarray, interval: int, sd: float, rng, cfg, warmup: int, dropout_p: float, days: int):
    """Sensor readings: minute indices and rounded, sensor-bounded values."""
    n = len(true_g)
    phase = int(rng.integers(0, interval))
    idx = np.arange(warmup + phase, n, interval)
    keep = np.ones(len(idx), bool)
    for day in range(days):
        if rng.random() < dropout_p:
            start = day * 1440 + rng.integers(0, 1440)
            length = rng.integers(10, 121)
            keep &= ~((idx >= start) & (idx < start + length))
    idx = idx[keep]
    vals = np.round(true_g[idx] + rng.normal(0, sd, len(idx)))
    vals = np.clip(vals, cfg["sensors"]["glucose_low_limit"], cfg["sensors"]["glucose_high_limit"])
    return idx, vals


def simulate_participant(
    pid: str,
    profile: pd.Series,
    days: int,
    start: pd.Timestamp,
    rng: np.random.Generator,
    cfg: dict,
    meal_library: list[dict],
    activity_cadence: int = 5,
    with_libre: bool = False,
) -> dict:
    sim = cfg["synthetic"]["simulation"]
    params = sim_params(profile, sim, rng)
    n = days * 1440
    minute = np.arange(n)

    # Sleep: a night before every day.
    asleep = np.zeros(n, bool)
    nights = []
    for k in range(days + 1):
        bed = (k - 1) * 1440 + int(rng.normal(22.75, 0.7) * 60)
        hours = float(np.clip(rng.uniform(5.0, 8.0), 4.0, 9.0))
        wake = bed + int(hours * 60)
        asleep[max(bed, 0) : max(min(wake, n), 0)] = True
        if 0 <= wake < n:
            nights.append((wake, hours))

    # Activity: awake ~1.2 METs, asleep ~0.95, plus walks and exercise by lifestyle.
    mets = np.where(asleep, 0.95, 1.2) + rng.normal(0, 0.05, n)
    lifestyle = rng.choice(["sedentary", "walker", "active"], p=[0.4, 0.4, 0.2])

    def bout(at: int, length: int, level: float) -> None:
        lo, hi = max(at, 0), min(at + length, n)
        if hi > lo:
            mets[lo:hi] = np.maximum(mets[lo:hi], level + rng.normal(0, 0.3, hi - lo))

    by_slot: dict[str, list[dict]] = {}
    for dish in meal_library:
        by_slot.setdefault(dish["slot"], []).append(dish)
    meals = []
    for day in range(days):
        base = day * 1440
        for slot in ("breakfast", "lunch", "dinner", "snack"):
            if slot not in by_slot or (slot == "snack" and rng.random() > 0.7):
                continue
            lo_h, hi_h = SLOT_HOURS[slot]
            at = base + int(rng.uniform(lo_h, hi_h) * 60)
            if at >= n or asleep[at]:
                continue
            dish = by_slot[slot][rng.integers(0, len(by_slot[slot]))]
            portion = float(np.exp(rng.normal(0, 0.2)))
            meals.append(
                {
                    "t": at,
                    "slot": slot,
                    "dish": dish["name"],
                    **{k: dish[k] * portion for k in ("carbs", "protein", "fat", "fiber", "kcal")},
                }
            )
            if slot in ("lunch", "dinner") and lifestyle in ("walker", "active") and rng.random() < 0.6:
                bout(at + int(rng.uniform(10, 30)), int(rng.uniform(15, 30)), 3.5)
        if lifestyle == "active" and rng.random() < 0.7:
            bout(base + int(rng.uniform(6.0, 7.0) * 60), 40, 5.0)
        if lifestyle == "sedentary":
            bout(base + int(rng.uniform(10, 18) * 60), 15, 3.2)
    meal_df = pd.DataFrame(meals).sort_values("t").reset_index(drop=True)

    # Glucose truth: baseline + meal responses + AR(1) physiological noise.
    rest = cfg["features"]["met_rest_level"]
    cum = np.concatenate([[0.0], np.cumsum(np.maximum(mets - rest, 0.0))])
    mt = meal_df["t"].to_numpy(int)
    e_m = cum[np.minimum(mt + 60, n)] - cum[mt]
    wake_t = np.array([w for w, _ in nights])
    wake_h = np.array([h for _, h in nights])
    k = np.searchsorted(wake_t, mt, side="right") - 1
    debt = np.where(k >= 0, np.maximum(cfg["sleep"]["reference_hours"] - wake_h[np.maximum(k, 0)], 0.0), 0.0)
    other = meal_df[["protein", "fat", "fiber"]].sum(axis=1).to_numpy()
    amp = (
        params["kc"]
        * meal_df["carbs"].to_numpy()
        / (1 + params["beta"] * other)
        * np.exp(-params["ka"] * e_m)
        * (1 + params["ks"] * debt)
    )
    S = minute[:, None] - mt[None, :]
    U = S / params["tau"]
    phi = np.where((S >= 0) & (S <= 1440), U * np.exp(1 - np.clip(U, None, 50)), 0.0)
    phi1 = sim["ar_phi_5min"] ** (1 / 5)
    eps = rng.normal(0, sim["ar_sd_mgdl"] * np.sqrt(1 - phi1**2), n)
    true_g = params["b0"] + phi @ amp + lfilter([1.0], [1.0, -phi1], eps)

    dex_idx, dex_val = _native_series(
        true_g,
        cfg["sensors"]["dexcom_interval_min"],
        sim["sensor_sd_mgdl"],
        rng,
        cfg,
        120,
        sim["dropout_prob_per_day"],
        days,
    )
    times = start + pd.to_timedelta(minute, unit="m")
    cgm = [pd.DataFrame({"pid": pid, "time": times[dex_idx], "glucose": dex_val, "sensor": "dexcom"})]
    libre = None
    if with_libre:
        lib_idx, lib_val = _native_series(
            true_g, cfg["sensors"]["libre_interval_min"], 6.0, rng, cfg, 60, sim["dropout_prob_per_day"], days
        )
        cgm.append(pd.DataFrame({"pid": pid, "time": times[lib_idx], "glucose": lib_val, "sensor": "libre"}))
        libre = (lib_idx, lib_val)
    cgm_df = pd.concat(cgm, ignore_index=True)
    cgm_df["censored"] = censor_flags(
        cgm_df["glucose"].to_numpy(float), cfg["sensors"]["glucose_low_limit"], cfg["sensors"]["glucose_high_limit"]
    )

    rhr = float(rng.normal(70, 6))
    hr = np.where(asleep, rhr - 6 + rng.normal(0, 2, n), rhr + 12 * (mets - 1) + rng.normal(0, 3, n))
    c = activity_cadence
    blocks = n // c
    act = pd.DataFrame(
        {
            "pid": pid,
            "time": times[np.arange(blocks) * c + c - 1],
            "hr": hr[: blocks * c].reshape(blocks, c).mean(axis=1).round(1),
            "mets": mets[: blocks * c].reshape(blocks, c).mean(axis=1).round(2),
            "dur_min": float(c),
        }
    )

    age = float(profile.get("age", 45))
    rmssd_mu = max(45 - 0.4 * (age - 30) - 0.3 * (rhr - 65), 12)
    sleep = pd.DataFrame(
        {
            "pid": pid,
            "night": [(start + pd.Timedelta(minutes=int(w))).normalize() for w, _ in nights],
            "sleep_hours": [round(h, 2) for _, h in nights],
            "coverage": 1.0,
            "available_time": [start + pd.Timedelta(minutes=int(w)) for w, _ in nights],
            "src": "synthetic",
            "rmssd": [round(float(np.exp(rng.normal(np.log(rmssd_mu), 0.25))), 1) for _ in nights],
        },
        columns=SLEEP_COLUMNS,
    )
    meals_out = pd.DataFrame(
        {
            "pid": pid,
            "time": start + pd.to_timedelta(meal_df["t"], unit="m"),
            "meal_type": meal_df["slot"],
            "dish": meal_df["dish"],
            "carbs": meal_df["carbs"].round(1),
            "protein": meal_df["protein"].round(1),
            "fat": meal_df["fat"].round(1),
            "fiber": meal_df["fiber"].round(1),
            "calories": meal_df["kcal"].round(0),
            "fraction": 1.0,
        }
    )
    for macro in ("carbs", "protein", "fat", "fiber"):
        meals_out[f"{macro}_eaten"] = meals_out[macro]
    return {
        "cgm": cgm_df,
        "activity": act,
        "meals": meals_out,
        "sleep": sleep,
        "truth": params | {"lifestyle": str(lifestyle), "rhr": rhr},
        "_minute_true": true_g,
        "_dexcom": (dex_idx, dex_val),
        "_libre": libre,
        "_mets": mets,
        "_hr": hr,
        "_meals_min": meal_df,
    }


def simulate_track(profiles: pd.DataFrame, cfg: dict, days: int | None = None, seed: int | None = None) -> TrackData:
    """The synthetic track: deterministic for a given seed (the dashboard regenerates it)."""
    syn = cfg["synthetic"]
    days = days or syn["days"]
    rng = np.random.default_rng(cfg["seed"] if seed is None else seed)
    prof = profiles.copy()
    if "pid" in prof.columns:
        prof = prof.set_index("pid")
    prof["homa_ir"] = prof["fasting_insulin"] * prof["fasting_glucose"] / 405.0
    prof["tg_hdl"] = prof["triglycerides"] / prof["hdl"]
    prof["included"] = True
    prof["exclusion_reason"] = ""
    library = syn["meal_library"]
    parts = {"cgm": [], "activity": [], "meals": [], "sleep": []}
    truth = {}
    for i, pid in enumerate(prof.index):
        start = pd.Timestamp(syn["reference_date"]) - pd.Timedelta(days=days + 1 + i % 7)
        out = simulate_participant(pid, prof.loc[pid], days, start, rng, cfg, library, syn["activity_cadence_min"])
        for key in parts:
            parts[key].append(out[key])
        truth[pid] = out["truth"]
    td = TrackData(
        "synthetic",
        prof,
        pd.concat(parts["cgm"], ignore_index=True),
        pd.concat(parts["activity"], ignore_index=True),
        pd.concat(parts["meals"], ignore_index=True),
        pd.concat(parts["sleep"], ignore_index=True),
    )
    from .align import wear_coverage

    dex = td.cgm[td.cgm["sensor"] == "dexcom"]
    interval = cfg["sensors"]["dexcom_interval_min"]
    td.profiles["primary_coverage"] = [
        wear_coverage(dex[dex["pid"] == pid]["time"], interval) for pid in td.profiles.index
    ]
    td.add_check(
        "Synthetic track",
        "info",
        "Every value is synthetic: Synthea EHRs plus simulated streams. Demo only, never evidence.",
    )
    td.meta.update({"clock_ok": True, "truth": truth, "track": "synthetic"})
    return td


def _represent(n: int, idx: np.ndarray, vals: np.ndarray, interval: int, mode: str) -> np.ndarray:
    """1-minute column for a sensor: blank, carried forward or interpolated between readings."""
    col = np.full(n, np.nan)
    if mode == "blank":
        col[idx] = vals
        return col
    for a, b, va, vb in zip(idx[:-1], idx[1:], vals[:-1], vals[1:], strict=True):
        if b - a > 1.5 * interval:  # dropout: leave the gap empty
            col[a] = va
            continue
        if mode == "carried":
            col[a:b] = va
        else:
            col[a:b] = va + (vb - va) * (np.arange(a, b) - a) / (b - a)
    if len(idx):
        col[idx[-1]] = vals[-1]
    return col


def write_cgmacros_like(
    out_dir: Path,
    cfg: dict,
    n_participants: int = 9,
    days: int = 3,
    seed: int = 7,
    dexcom_mode: str = "blank",
    libre_mode: str = "interpolated",
) -> dict[str, int]:
    """Write bio.csv and CGMacros-0XX/CGMacros-0XX.csv files shaped like the real dataset."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    groups = ["healthy", "prediabetes", "t2d"]
    hb = {"healthy": (5.2, 0.2, 4.6, 5.6), "prediabetes": (6.0, 0.2, 5.7, 6.4), "t2d": (7.3, 0.6, 6.5, 8.5)}
    fg = {"healthy": 90, "prediabetes": 104, "t2d": 140}
    bio_rows = []
    for i in range(n_participants):
        grp = groups[i % 3]
        mu, sd, lo, hi = hb[grp]
        number = 2 * i + 1  # ids with gaps, like the real folders
        bio_rows.append(
            {
                "subject": number,
                "Age": int(rng.integers(25, 69)),
                "Gender": rng.choice(["F", "M"]),
                "BMI": round(float(rng.uniform(21, 40)), 2),
                "Body weight": round(float(rng.uniform(120, 260)), 1),
                "Height": round(float(rng.uniform(60, 72)), 1),
                "Self-identify": "Not reported",
                "A1c PDL (Lab)": round(float(np.clip(rng.normal(mu, sd), lo, hi)), 1),
                "Fasting GLU - PDL (Lab)": round(float(rng.normal(fg[grp], 6)), 0),
                "Insulin": round(float(rng.uniform(4, 30)), 1),
                "Triglycerides": round(float(rng.uniform(60, 250)), 0),
                "Cholesterol": round(float(rng.uniform(140, 260)), 0),
                "HDL": round(float(rng.uniform(35, 80)), 0),
                "Non HDL": round(float(rng.uniform(90, 190)), 0),
                "LDL (Cal)": 800 if i == 1 else round(float(rng.uniform(70, 170)), 0),
                "VLDL (Cal)": 400 if i == 2 else round(float(rng.uniform(10, 50)), 0),
                "_group": grp,
            }
        )
    bio = pd.DataFrame(bio_rows)
    for row in bio_rows:
        pid = f"CGM-{row['subject']:03d}"
        profile = pd.Series(
            {"hba1c_pct": row["A1c PDL (Lab)"], "fasting_glucose": row["Fasting GLU - PDL (Lab)"], "age": row["Age"]}
        )
        start = pd.Timestamp("2023-03-01") + pd.Timedelta(days=int(rng.integers(0, 400)))
        sim = simulate_participant(
            pid, profile, days, start, rng, cfg, FIXTURE_MEALS, activity_cadence=1, with_libre=True
        )
        n = days * 1440
        minute_times = start + pd.to_timedelta(np.arange(n), unit="m")
        dex_idx, dex_val = sim["_dexcom"]
        lib_idx, lib_val = sim["_libre"]
        frame = pd.DataFrame(
            {
                "Timestamp": minute_times.strftime("%Y-%m-%d %H:%M:%S"),
                "Libre GL": _represent(n, lib_idx, lib_val, cfg["sensors"]["libre_interval_min"], libre_mode),
                "Dexcom GL": _represent(n, dex_idx, dex_val, cfg["sensors"]["dexcom_interval_min"], dexcom_mode),
                "HR": np.round(sim["_hr"], 0),
                "Calories (Activity)": np.round(sim["_mets"] * 1.2, 3),
                "METs": np.round(sim["_mets"] * 10).astype(int),
                "Meal Type": None,
                "Calories": np.nan,
                "Carbs": np.nan,
                "Protein": np.nan,
                "Fat": np.nan,
                "Fiber": np.nan,
                "Amount Consumed": np.nan,
                "Image path": None,
            }
        )
        for _, meal in sim["_meals_min"].iterrows():
            r = int(meal["t"])
            frame.loc[r, "Meal Type"] = meal["slot"].capitalize()
            frame.loc[r, "Calories"] = round(meal["kcal"])
            frame.loc[r, "Carbs"] = round(meal["carbs"], 1)
            frame.loc[r, "Protein"] = round(meal["protein"], 1)
            frame.loc[r, "Fat"] = round(meal["fat"], 1)
            frame.loc[r, "Fiber"] = round(meal["fiber"], 1)
            frame.loc[r, "Amount Consumed"] = int(rng.choice([50, 75, 100, 100, 100]))
            frame.loc[r, "Image path"] = f"photos/{r:05d}.jpg"
        folder = out_dir / f"CGMacros-{row['subject']:03d}"
        folder.mkdir(exist_ok=True)
        frame.to_csv(folder / f"CGMacros-{row['subject']:03d}.csv", index=False)
    bio.drop(columns="_group").to_csv(out_dir / "bio.csv", index=False)
    return bio["_group"].value_counts().to_dict()
