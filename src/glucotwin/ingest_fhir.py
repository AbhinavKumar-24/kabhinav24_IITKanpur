"""Synthetic track, step 1: Synthea FHIR R4 bundles -> one compact profile row per patient.

Run through ``scripts/generate_synthea.sh`` or directly::

    python -m glucotwin.ingest_fhir --fhir-dir data/synthetic/synthea_raw/output/fhir --out data/synthetic

Identifiers (names, addresses, telecom) are never read into the profile table.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config

LOINC = {
    "hba1c_pct": ("4548-4",),
    "fasting_glucose": ("2339-0", "2345-7"),
    "bmi": ("39156-5",),
    "triglycerides": ("2571-8",),
    "hdl": ("2085-9",),
    "total_chol": ("2093-3",),
    "ldl": ("18262-6",),
    "sbp": ("8480-6",),
    "dbp": ("8462-4",),
}
SNOMED = {
    "t2d": ("44054006",),
    "prediabetes": ("714628002", "15777000"),
    "hypertension": ("59621000",),
    "hyperlipidemia": ("55822004",),
    "metabolic_syndrome": ("237602007",),
}
DIABETES_DRUG_WORDS = ("metformin", "insulin", "glipizide", "glimepiride", "liraglutide", "sitagliptin")
KEEP_SAMPLE_TYPES = {"Patient", "Condition", "Observation", "MedicationRequest"}


def _codes(concept: dict | None) -> list[str]:
    if not concept:
        return []
    return [c.get("code", "") for c in concept.get("coding", [])]


def _display(concept: dict | None) -> str:
    if not concept:
        return ""
    for c in concept.get("coding", []):
        if c.get("display"):
            return c["display"]
    return concept.get("text", "")


def _status(resource: dict) -> str:
    coding = (resource.get("clinicalStatus") or {}).get("coding") or [{}]
    return coding[0].get("code", "")


def parse_bundle(path: Path, reference: date) -> dict | None:
    """Parse one Synthea bundle into a flat record; returns None for deceased patients."""
    bundle = json.loads(Path(path).read_text(encoding="utf-8"))
    latest: dict[str, tuple[str, float]] = {}
    conditions: set[str] = set()
    condition_names: set[str] = set()
    drugs: set[str] = set()
    patient = None
    for entry in bundle.get("entry", []):
        res = entry.get("resource", {})
        rtype = res.get("resourceType")
        if rtype == "Patient":
            patient = res
        elif rtype == "Observation":
            obs_time = res.get("effectiveDateTime", "")
            pairs = []
            if "valueQuantity" in res:
                pairs.append((_codes(res.get("code")), res["valueQuantity"].get("value")))
            for comp in res.get("component", []):
                if "valueQuantity" in comp:
                    pairs.append((_codes(comp.get("code")), comp["valueQuantity"].get("value")))
            for codes, value in pairs:
                if value is None:
                    continue
                for field_name, loincs in LOINC.items():
                    if any(c in loincs for c in codes):
                        if field_name not in latest or obs_time >= latest[field_name][0]:
                            latest[field_name] = (obs_time, float(value))
        elif rtype == "Condition":
            if _status(res) not in ("active", "recurrence", "relapse", ""):
                continue
            codes = _codes(res.get("code"))
            for key, snomed in SNOMED.items():
                if any(c in snomed for c in codes):
                    conditions.add(key)
            condition_names.add(_display(res.get("code")))
        elif rtype == "MedicationRequest":
            if res.get("status") not in ("active", "completed", None):
                continue
            name = _display(res.get("medicationCodeableConcept")).lower()
            for word in DIABETES_DRUG_WORDS:
                if word in name:
                    drugs.add(word)
    if patient is None or patient.get("deceasedDateTime") or patient.get("deceasedBoolean"):
        return None
    birth = date.fromisoformat(patient["birthDate"])
    age = reference.year - birth.year - ((reference.month, reference.day) < (birth.month, birth.day))
    if "t2d" in conditions:
        group = "t2d"
    elif "prediabetes" in conditions:
        group = "prediabetes"
    else:
        group = "healthy"
    record = {
        "fhir_id": patient["id"],
        "source_bundle": Path(path).name.split("_")[-1],  # uuid part only, no names
        "age": age,
        "sex": {"male": "M", "female": "F"}.get(patient.get("gender", ""), "U"),
        "group": group,
        "conditions": ";".join(sorted(conditions)),
        "metformin": "metformin" in drugs,
        "insulin_therapy": "insulin" in drugs,
    }
    for field_name in LOINC:
        record[field_name] = latest[field_name][1] if field_name in latest else np.nan
    return record


def _simulate_missing(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Fill labs Synthea did not produce, from group-specific distributions (tagged simulated)."""
    hba1c = {"healthy": (5.3, 0.2, 4.6, 5.6), "prediabetes": (6.0, 0.2, 5.7, 6.4), "t2d": (7.4, 0.8, 6.5, 11.0)}
    glucose = {"healthy": (92, 7, 75, 99), "prediabetes": (108, 7, 100, 125), "t2d": (145, 25, 126, 260)}
    plausible = {"hba1c_pct": (4.0, 14.0), "fasting_glucose": (60.0, 400.0)}
    for col, table in (("hba1c_pct", hba1c), ("fasting_glucose", glucose)):
        lo_ok, hi_ok = plausible[col]
        df.loc[~df[col].between(lo_ok, hi_ok), col] = np.nan  # implausible Synthea values
        prov = np.where(df[col].isna(), "simulated", "synthea")
        for grp, (mu, sd, lo, hi) in table.items():
            mask = df[col].isna() & (df["group"] == grp)
            df.loc[mask, col] = np.clip(rng.normal(mu, sd, mask.sum()), lo, hi).round(1)
        df["prov_" + col] = prov
    bmi_missing = df["bmi"].isna()
    df.loc[bmi_missing, "bmi"] = np.clip(rng.normal(27, 4, bmi_missing.sum()), 18, 45).round(1)
    df["prov_bmi"] = np.where(bmi_missing, "simulated", "synthea")
    # Synthea does not model fasting insulin: always simulated from BMI and group.
    log_ins = (
        np.log(8.0)
        + 0.04 * (df["bmi"] - 25.0)
        + df["group"].map({"healthy": 0.0, "prediabetes": 0.15, "t2d": 0.35})
        + rng.normal(0, 0.3, len(df))
    )
    df["fasting_insulin"] = np.clip(np.exp(log_ins), 2, 60).round(1)
    df["prov_fasting_insulin"] = "simulated"
    return df


def build_profiles(fhir_dir: Path, cfg: dict, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    """Parse every bundle, keep living adults aged 25-75, select the target mix."""
    syn = cfg["synthetic"]
    reference = date.fromisoformat(syn["reference_date"])
    files = sorted(Path(fhir_dir).glob("*.json"))
    files = [f for f in files if not f.name.startswith(("hospitalInformation", "practitionerInformation"))]
    records = [r for r in (parse_bundle(f, reference) for f in files) if r is not None]
    all_df = pd.DataFrame(records)
    eligible = all_df[(all_df["age"] >= 25) & (all_df["age"] <= 75)].reset_index(drop=True)

    rng = np.random.default_rng(seed)
    bands = {}
    for label in syn["age_mix"]:
        lo, hi = (int(x) for x in label.split("-"))
        bands[label] = (lo, hi)

    def band_of(age: int) -> str:
        for label, (lo, hi) in bands.items():
            if lo <= age <= hi:
                return label
        return "other"

    eligible["age_band"] = eligible["age"].map(band_of)
    chosen = []
    shortfall = {}
    for group, target in syn["mix"].items():
        pool = eligible[eligible["group"] == group]
        if len(pool) <= target:
            chosen.append(pool)
            shortfall[group] = int(target - len(pool))
            continue
        share = pool["age_band"].value_counts(normalize=True)
        weights = pool["age_band"].map(lambda b, sh=share: syn["age_mix"].get(b, 0.0) / max(sh.get(b, 1.0), 1e-9))
        weights = (weights / weights.sum()).to_numpy()
        idx = rng.choice(len(pool), size=target, replace=False, p=weights)
        chosen.append(pool.iloc[np.sort(idx)])
    prof = pd.concat(chosen).reset_index(drop=True)
    prof = _simulate_missing(prof, rng)
    prof["tcf7l2_risk_alleles"] = rng.binomial(2, syn["tcf7l2_risk_allele_freq"], len(prof))
    prof["prov_tcf7l2_risk_alleles"] = "simulated"
    prof["pid"] = "SYN-" + prof["fhir_id"].str.replace("-", "").str[:8]
    if prof["pid"].duplicated().any():  # fall back to longer ids on the rare collision
        prof["pid"] = "SYN-" + prof["fhir_id"].str.replace("-", "").str[:16]
    prof = prof.drop(columns=["fhir_id", "age_band"])
    lead = ["pid", "group", "age", "sex", "bmi", "hba1c_pct", "fasting_glucose", "fasting_insulin"]
    prof = prof[lead + [c for c in prof.columns if c not in lead]]
    summary = {
        "bundles_read": len(files),
        "alive_patients": len(all_df),
        "eligible_age_25_75": int(len(eligible)),
        "eligible_by_group": eligible["group"].value_counts().to_dict(),
        "selected_by_group": prof["group"].value_counts().to_dict(),
        "shortfall_by_group": shortfall,
        "hba1c_simulated": int((prof["prov_hba1c_pct"] == "simulated").sum()),
        "fasting_glucose_simulated": int((prof["prov_fasting_glucose"] == "simulated").sum()),
    }
    return prof, summary


def trim_bundle(path: Path, out_path: Path) -> None:
    """Keep only the resource types the profile uses, without names, addresses or telecom."""
    bundle = json.loads(Path(path).read_text(encoding="utf-8"))
    keep = []
    per_code: dict[str, int] = {}
    entries = sorted(
        bundle.get("entry", []),
        key=lambda e: e.get("resource", {}).get("effectiveDateTime") or e.get("resource", {}).get("authoredOn") or "",
        reverse=True,
    )
    for entry in entries:
        res = entry.get("resource", {})
        rtype = res.get("resourceType")
        if rtype not in KEEP_SAMPLE_TYPES:
            continue
        if rtype == "Patient":
            res = {
                k: v
                for k, v in res.items()
                if k
                not in {
                    "name",
                    "address",
                    "telecom",
                    "identifier",
                    "text",
                    "extension",
                    "communication",
                    "maritalStatus",
                }
            }
        if rtype == "Observation":
            codes = _codes(res.get("code"))
            comp_codes = [c for comp in res.get("component", []) for c in _codes(comp.get("code"))]
            wanted = {c for v in LOINC.values() for c in v}
            if not (wanted & set(codes + comp_codes)):
                continue
            key = (codes or ["?"])[0]
            per_code[key] = per_code.get(key, 0) + 1
            if per_code[key] > 3:  # the three most recent results per test are enough
                continue
        if rtype == "MedicationRequest":
            per_code["_med"] = per_code.get("_med", 0) + 1
            if per_code["_med"] > 5:
                continue
        res.pop("text", None)
        keep.append({"fullUrl": entry.get("fullUrl"), "resource": res})
    trimmed = {"resourceType": "Bundle", "type": bundle.get("type", "transaction"), "entry": keep}
    out_path.write_text(json.dumps(trimmed, indent=1), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fhir-dir", required=True)
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--config", default=None)
    ap.add_argument("--synthea-version", default="unknown")
    ap.add_argument("--population", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reference-date", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prof, summary = build_profiles(Path(args.fhir_dir), cfg, seed=args.seed)
    prof.to_csv(out / "synthetic_profiles.csv", index=False)

    sample_dir = out / "fhir_sample"
    sample_dir.mkdir(exist_ok=True)
    for old in sample_dir.glob("*.json"):
        old.unlink()
    for group in ("healthy", "prediabetes", "t2d"):
        row = prof[prof["group"] == group].head(1)
        if row.empty:
            continue
        match = sorted(Path(args.fhir_dir).glob(f"*_{row['source_bundle'].iloc[0]}"))
        if match:
            trim_bundle(match[0], sample_dir / f"sample_{group}.json")

    summary.update(
        {
            "synthea_version": args.synthea_version,
            "population": args.population,
            "seed": args.seed,
            "reference_date": args.reference_date or cfg["synthetic"]["reference_date"],
            "command": "bash scripts/generate_synthea.sh",
            "note": "Synthetic records only. HbA1c and fasting glucose missing from Synthea, and all "
            "fasting insulin and genetic-marker values, are simulated (see prov_* columns).",
        }
    )
    (out / "synthea_run.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
