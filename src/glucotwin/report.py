"""Reports written by the pipeline. Scores are never edited by hand."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .models import MODEL_NAMES

SYNTHETIC_BANNER = (
    "> **SYNTHETIC DEMO.** Every value here comes from simulated data generated with a meal-response "
    "model like the twin itself. These numbers show the pipeline runs end to end; they are **not** "
    "evidence of accuracy."
)
EVIDENCE_NOTE = (
    "> Exploratory feasibility results on a small public cohort (CGMacros). Not clinical validation, "
    "and not evidence of performance in Indian patients."
)


def _f(x, digits: int = 2, pct: bool = False) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{100 * x:.0f}%" if pct else f"{x:.{digits}f}"


def ci(v, digits: int = 2, pct: bool = False) -> str:
    if v is None or len(v) != 3:
        return "–"
    est, lo, hi = v
    if est is None or (isinstance(est, float) and math.isnan(est)):
        return "–"
    if lo is None or (isinstance(lo, float) and math.isnan(lo)):
        return _f(est, digits, pct)
    return f"{_f(est, digits, pct)} ({_f(lo, digits, pct)}–{_f(hi, digits, pct)})"


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, default=_json_default, allow_nan=True).replace("NaN", "null")
    path.write_text(text, encoding="utf-8")


def data_summary(track: str, checks, desc: dict) -> str:
    lines = [f"# Data summary — {track} track", ""]
    if track == "synthetic":
        lines += [SYNTHETIC_BANNER, ""]
    lines += [f"Generated {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')} by `scripts/run_pipeline.py`.", ""]
    lines += ["## Validation checks", "", "| Check | Status | Detail |", "|---|---|---|"]
    for c in checks:
        lines.append(f"| {c.name} | {c.status} | {c.detail} |")
    ep = desc["events_per_participant"]
    lines += [
        "",
        "## Counts",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Participants in analysis / total | {desc['participants_in_analysis']} / {desc['participants_total']} |",
        f"| Groups | {desc['groups']} |",
        f"| Days of CGM | {desc['days_of_cgm']} |",
        f"| Mean primary-sensor coverage (observed / expected readings) | {_f(desc['primary_coverage_mean'], pct=True)} |",
        f"| Imputed share of timeline rows | {_f(desc['imputed_share'], pct=True)} |",
        f"| Censored readings (<= 40 / >= 400) | {desc['censored_low']} / {desc['censored_high']} |",
        f"| Eligible prediction moments | {desc['eligible_moments']} |",
        f"| Evaluable / unscorable windows | {desc['evaluable_windows']} / {desc['unscorable_windows']} |",
        f"| Confirmed events (total; median, min–max per participant) | {desc['events_total']}; {ep['median']:.0f}, {ep['min']}–{ep['max']} |",
    ]
    if desc["participants_excluded"]:
        lines += ["", "## Excluded participants", "", "| Participant | Reason |", "|---|---|"]
        lines += [f"| {pid} | {why} |" for pid, why in desc["participants_excluded"].items()]
    return "\n".join(lines) + "\n"


def results_md(track: str, res: dict, desc: dict, cfg: dict) -> str:
    L = [f"# Results — {track} track", "", SYNTHETIC_BANNER if track == "synthetic" else EVIDENCE_NOTE, ""]
    L += [
        f"Nested participant-wise cross-validation: {cfg['cv']['outer_folds']} outer folds (reporting only), "
        f"{cfg['cv']['inner_folds']} inner folds (all selection). 95% intervals from {cfg['cv']['bootstrap']} "
        f"bootstrap resamples of participants. {desc['participants_in_analysis']} participants, "
        f"{desc['events_total']} confirmed events, {desc['evaluable_windows']} evaluable and "
        f"{desc['unscorable_windows']} unscorable windows.",
        "",
        "## Event prediction (outer folds)",
        "",
        "Events warned early = an alert 30–120 minutes before onset. Lead time is measured to the onset. "
        f"Alert budget {cfg['alerts']['budget_per_day']} per patient-day, {cfg['alerts']['cooldown_min']}-minute cooldown.",
        "",
        "| Model | Warned early | Warned late | Missed | Median lead (min) | False alerts / patient-day | AUPRC | AUROC | Events | Alerts (unscorable) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for m, r in res["models"].items():
        L.append(
            f"| {m} {MODEL_NAMES[m]} | {ci(r['early_rate'], pct=True)} | {ci(r['late_rate'], pct=True)} | {ci(r['miss_rate'], pct=True)} | "
            f"{ci(r['median_lead_min'], 0)} | {ci(r['false_alerts_per_day'])} | {ci(r['auprc'])} | {ci(r['auroc'])} | "
            f"{r['events']} | {r['alerts']} ({r['unscorable_alerts']}) |"
        )
    L += [
        "",
        "**Selection inside each outer fold (inner cross-validation):**",
        "",
        "| Fold | Dashboard model | Twin extensions |",
        "|---|---|---|",
    ]
    for s in res["selection"]:
        L.append(f"| {s['fold']} | {s['model']} | {', '.join(s['extensions']) or 'none'} |")
    if "extensions" in res:
        L += [
            "",
            "## Twin extensions (all reported, selected or not)",
            "",
            "Change in 60-minute forecast RMSE (extension minus base), twin fitted on each held-out participant's first 72 hours.",
            "",
            "| Extension | Selected in folds | RMSE change, mg/dL (95% CI) | Participants |",
            "|---|---|---|---|",
        ]
        names = {
            "E1": "E1 protein/fat/fibre damping",
            "E2": "E2 observed post-meal activity",
            "E3": "E3 estimated sleep",
        }
        for e, r in res["extensions"].items():
            L.append(
                f"| {names.get(e, e)} | {r['selected_in_folds']} of {len(res['selection'])} | {ci(r['rmse_diff_mgdl'])} | {r['participants']} |"
            )
    if "personalisation" in res:
        hrs = cfg["twin"]["personalisation_hours"]
        L += [
            "",
            "## Personalisation (cold start)",
            "",
            f"Twin 60-minute forecast RMSE (mg/dL) on each held-out participant's data from hour {cfg['twin']['evaluation_from_hour']} on. The 0 h column is the prior alone; later columns add the participant's own first hours.",
            "",
            "| Starting point | " + " | ".join(f"{h} h" for h in hrs) + " |",
            "|---|" + "---|" * len(hrs),
        ]
        for label, title in (("P0", "Population default (P0)"), ("P1", "Profile-informed prior (P1 → P2)")):
            row = res["personalisation"][label]
            L.append(f"| {title} | " + " | ".join(ci(row[str(h)]["rmse"], 1) for h in hrs) + " |")
    if "forecast" in res:
        L += [
            "",
            "## Glucose forecast (secondary)",
            "",
            "| Method | Horizon | RMSE mg/dL (95% CI) | MAE mg/dL (95% CI) | Clarke A+B | Points |",
            "|---|---|---|---|---|---|",
        ]
        for name in ("twin", "trend"):
            for h, r in res["forecast"][name].items():
                L.append(
                    f"| {'Personal twin (anchored)' if name == 'twin' else 'Trend baseline'} | +{h} min | {ci(r['rmse'], 1)} | {ci(r['mae'], 1)} | {_f(r['clarke_ab_pct'] / 100 if r['clarke_ab_pct'] == r['clarke_ab_pct'] else float('nan'), pct=True)} | {r['points']} |"
                )
    if "propensity" in res:
        L += [
            "",
            "## Baseline propensity (profile only)",
            "",
            "A lab profile cannot time an excursion; this relates HbA1c to each participant's overall excursion rate.",
            "",
            f"Spearman correlation, HbA1c vs confirmed events per day: {ci(res['propensity']['spearman_hba1c_vs_events_per_day'])} ({res['propensity']['participants']} participants).",
        ]
    if "identifiability" in res:
        L += [
            "",
            "## Twin parameter spread (identifiability check)",
            "",
            "Median standard deviation of each parameter when the base twin is refitted on bootstrap resamples of a participant's days. Parameters are descriptive, not physiological.",
            "",
            "| Parameter | Median SD | Participants |",
            "|---|---|---|",
        ]
        for k, v in res["identifiability"].items():
            L.append(f"| {k} | {_f(v['median_sd'], 3)} | {v['participants']} |")
    L += [
        "",
        "## By diagnosis group (descriptive only)",
        "",
        "| Model | Group | Warned early | False alerts / patient-day | Participants |",
        "|---|---|---|---|---|",
    ]
    for m, r in res["models"].items():
        for g, v in r["by_group"].items():
            L.append(
                f"| {m} | {g} | {_f(v['early_rate'], pct=True)} | {_f(v['false_alerts_per_day'])} | {v['participants']} |"
            )
    if "sensitivity" in res:
        L += [
            "",
            "## Sensitivity analyses (M0–M2)",
            "",
            "| Variant | Model | Warned early | False alerts / patient-day | Events | Participants |",
            "|---|---|---|---|---|---|",
        ]
        for variant, block in res["sensitivity"].items():
            for m, r in block["models"].items():
                L.append(
                    f"| {variant} | {m} | {ci(r['early_rate'], pct=True)} | {ci(r['false_alerts_per_day'])} | {r['events']} | {block['participants']} |"
                )
    else:
        L += ["", "Sensitivity analyses were not run (use `--sensitivity`)."]
    return "\n".join(L) + "\n"
