"""Event models M0-M3 (APPROACH.md §5). Hyperparameters are fixed in config.yaml."""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import model_features

MODELS = ("M0", "M1", "M2", "M3")
MODEL_NAMES = {
    "M0": "Trend baseline",
    "M1": "Dynamic (LightGBM)",
    "M2": "Fused (LightGBM)",
    "M3": "Twin-enhanced (LightGBM)",
}


def m0_score(X: pd.DataFrame) -> np.ndarray:
    """Projected glucose at +60 min from the last 30 minutes' trend."""
    return (X["g_now"] + 60.0 * X["slope_30"].fillna(0.0)).to_numpy(float)


@dataclass
class Fitted:
    name: str
    features: list[str]
    booster: lgb.Booster | None = None
    constant: float | None = None


def train(name: str, X: pd.DataFrame, y: np.ndarray, cfg: dict, clock_ok: bool) -> Fitted:
    feats = model_features(name, clock_ok)
    if name == "M0":
        return Fitted(name, feats)
    yv = np.asarray(y).astype(int)
    if len(np.unique(yv)) < 2:  # nothing to learn from a single class
        return Fitted(name, feats, constant=float(yv.mean()) if len(yv) else 0.0)
    clf = lgb.LGBMClassifier(**cfg["lightgbm"])
    clf.fit(X[feats], yv)
    return Fitted(name, feats, booster=clf.booster_)


def score(fitted: Fitted, X: pd.DataFrame) -> np.ndarray:
    if fitted.name == "M0":
        return m0_score(X)
    if fitted.booster is None:
        return np.full(len(X), fitted.constant if fitted.constant is not None else 0.0)
    return np.asarray(fitted.booster.predict(X[fitted.features]), dtype=float)
