"""Plain-language reasons from TreeSHAP contributions (APPROACH.md §8).

LightGBM computes exact TreeSHAP values itself (``pred_contrib=True``). Contributions are summed
by feature family. Reasons are factors the model associated with higher risk, never causes.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .features import FAMILIES
from .models import Fitted

FAMILY_OF = {f: fam for fam, feats in FAMILIES.items() for f in feats}


def contributions(fitted: Fitted, x_row: pd.DataFrame) -> pd.Series | None:
    if fitted.booster is None:
        return None
    raw = fitted.booster.predict(x_row[fitted.features], pred_contrib=True)[0]
    return pd.Series(raw[:-1], index=fitted.features)


def _sentence(family: str, x: pd.Series, cap: float = 360.0) -> str:
    if family == "glucose":
        s = x.get("slope_30", np.nan)
        trend = "steady" if not np.isfinite(s) or abs(s) < 0.05 else ("rising" if s > 0 else "falling")
        rate = f" {abs(s):.1f} mg/dL per minute" if np.isfinite(s) and abs(s) >= 0.05 else ""
        return f"Glucose is {x['g_now']:.0f} mg/dL and {trend}{rate} over the last 30 minutes."
    if family == "meals":
        if x["mins_since_meal"] < cap:
            return (
                f"Last meal {x['mins_since_meal']:.0f} minutes ago had {x['last_meal_carbs']:.0f} g of "
                f"carbohydrate ({x['carbs_on_board']:.0f} g still counted as on board)."
            )
        return "No meal logged in the last 6 hours."
    if family == "activity":
        mets = x.get("mets_30", np.nan)
        part = (
            f"Activity in the last 30 minutes averaged {mets:.1f} METs"
            if np.isfinite(mets)
            else "No recent activity data"
        )
        return part + f"; {x.get('met_min_since_meal', 0):.0f} MET-minutes of movement since the last meal."
    if family == "time":
        hour = (math.degrees(math.atan2(x["hour_sin"], x["hour_cos"])) / 15.0) % 24
        return f"Time of day ({int(hour):02d}:{int((hour % 1) * 60):02d})."
    if family == "profile":
        return (
            f"Lab profile: HbA1c {x['hba1c_pct']:.1f}%, fasting glucose {x['fasting_glucose']:.0f} mg/dL, "
            f"BMI {x['bmi']:.1f}."
        )
    if family == "twin":
        return (
            f"The personal twin forecasts {x['twin_f60']:.0f} mg/dL in 60 minutes and a peak of "
            f"{x['twin_peak']:.0f} mg/dL within 2 hours."
        )
    return family


def reasons(fitted: Fitted, x_row: pd.DataFrame, top: int = 3) -> list[dict]:
    """Top families raising the risk, plus the one lowering it most."""
    x = x_row.iloc[0]
    contrib = contributions(fitted, x_row)
    if contrib is None:
        s = float(np.nan_to_num(x.get("slope_30", 0.0)))
        proj = x["g_now"] + 60 * s
        return [
            {
                "family": "glucose",
                "effect": "raises risk" if s > 0 else "lowers risk",
                "weight": float("nan"),
                "text": f"Trend baseline: glucose {x['g_now']:.0f} mg/dL, changing {s:+.1f} mg/dL per minute, projects {proj:.0f} mg/dL in 60 minutes.",
            }
        ]
    fam = contrib.groupby(contrib.index.map(FAMILY_OF)).sum()
    out = []
    for name, w in fam[fam > 0].sort_values(ascending=False).head(top).items():
        out.append({"family": name, "effect": "raises risk", "weight": float(w), "text": _sentence(name, x)})
    for name, w in fam[fam < 0].sort_values().head(1).items():
        out.append({"family": name, "effect": "lowers risk", "weight": float(w), "text": _sentence(name, x)})
    return out
