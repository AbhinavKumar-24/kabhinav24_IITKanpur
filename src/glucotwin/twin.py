"""Personal twin: a three-parameter meal-response model with optional extensions (APPROACH.md §4).

    G_hat(t) = b0 + sum over meals m with t_m <= t of  kc * C_m * phi(t - t_m; tau)
    phi(s; tau) = (s / tau) * exp(1 - s / tau)

The model is phenomenological: it does not simulate insulin, digestion or medication, and its
parameters are descriptive. Extensions E1 (protein/fat/fibre damping), E2 (observed post-meal
activity) and E3 (estimated sleep) are only used if inner cross-validation selects them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.linear_model import Ridge

from .schema import PROFILE_FEATURES

BASE = ("b0", "log_kc", "log_tau")
EXT_PARAM = {"E1": "log_beta", "E2": "log_ka", "E3": "ks"}
GENERIC = {
    "b0": (100.0, 40.0),
    "log_kc": (math.log(0.8), 1.0),
    "log_tau": (math.log(55.0), 0.5),
    "log_beta": (math.log(0.003), 1.5),
    "log_ka": (math.log(0.005), 1.5),
    "ks": (0.0, 0.2),
}
BOUNDS = {
    "b0": (40.0, 300.0),
    "log_kc": (math.log(0.02), math.log(8.0)),
    "log_tau": (math.log(15.0), math.log(180.0)),
    "log_beta": (math.log(1e-5), math.log(0.2)),
    "log_ka": (math.log(1e-5), math.log(0.2)),
    "ks": (-0.5, 1.0),
}
SD_FLOOR = {"b0": 5.0, "log_kc": 0.1, "log_tau": 0.05, "log_beta": 0.3, "log_ka": 0.3, "ks": 0.05}


@dataclass(frozen=True)
class TwinSpec:
    extensions: tuple[str, ...] = ()

    @property
    def names(self) -> list[str]:
        return list(BASE) + [EXT_PARAM[e] for e in self.extensions]

    def has(self, ext: str) -> bool:
        return ext in self.extensions

    @property
    def label(self) -> str:
        return "base" + "".join("+" + e for e in self.extensions)

    def generic_prior(self) -> tuple[np.ndarray, np.ndarray]:
        return (np.array([GENERIC[n][0] for n in self.names]), np.array([GENERIC[n][1] for n in self.names]))

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return (np.array([BOUNDS[n][0] for n in self.names]), np.array([BOUNDS[n][1] for n in self.names]))

    def clip(self, theta: np.ndarray) -> np.ndarray:
        lo, hi = self.bounds()
        return np.clip(theta, lo + 1e-9, hi - 1e-9)

    def readable(self, theta: np.ndarray) -> dict[str, float]:
        p = dict(zip(self.names, theta, strict=True))
        out = {
            "baseline_mgdl": p["b0"],
            "rise_per_g_carb": math.exp(p["log_kc"]),
            "time_to_peak_min": math.exp(p["log_tau"]),
        }
        if "log_beta" in p:
            out["damping_per_g"] = math.exp(p["log_beta"])
        if "log_ka" in p:
            out["activity_effect"] = math.exp(p["log_ka"])
        if "ks" in p:
            out["sleep_effect"] = p["ks"]
        return out


@dataclass
class TwinInputs:
    """One participant's meals and activity in minutes since that participant's start."""

    meal_t: np.ndarray
    carbs: np.ndarray
    other: np.ndarray  # protein + fat + fibre eaten (E1)
    debt: np.ndarray  # sleep debt (hours) known at meal time (E3)
    act_t: np.ndarray
    act_cum: np.ndarray  # cumulative MET-minutes above the rest level (E2)
    memory: float = 1440.0
    act_window: float = 60.0

    def cum_excess(self, x: np.ndarray) -> np.ndarray:
        if len(self.act_t) == 0:
            return np.zeros_like(np.asarray(x, dtype=float))
        idx = np.searchsorted(self.act_t, x, side="right")
        return np.where(idx > 0, self.act_cum[np.maximum(idx - 1, 0)], 0.0)

    def window(self, lo: float, hi: float) -> TwinInputs:
        keep = (self.meal_t >= lo) & (self.meal_t <= hi)
        return TwinInputs(
            self.meal_t[keep],
            self.carbs[keep],
            self.other[keep],
            self.debt[keep],
            self.act_t,
            self.act_cum,
            self.memory,
            self.act_window,
        )


def _design(inputs: TwinInputs, x: np.ndarray, until: np.ndarray, need_e: bool):
    """Elapsed-time matrix, mask and E2 exposure for evaluation times ``x``.

    ``until`` is the latest time whose data may be used for each evaluation (meals after it are
    unknown; activity after it is not counted).
    """
    x = np.asarray(x, dtype=float)
    until = np.broadcast_to(np.asarray(until, dtype=float), x.shape)
    mt = inputs.meal_t
    S = x[:, None] - mt[None, :]
    mask = (S >= 0) & (S <= inputs.memory) & (mt[None, :] <= until[:, None])
    E = None
    if need_e:
        upper = np.minimum(np.minimum(x[:, None], until[:, None]), mt[None, :] + inputs.act_window)
        E = inputs.cum_excess(upper) - inputs.cum_excess(mt)[None, :]
        E = np.where(mask, np.maximum(E, 0.0), 0.0)
    return S, mask, E


def _curve(theta: np.ndarray, spec: TwinSpec, inputs: TwinInputs, S, mask, E, jac: bool = False):
    p = dict(zip(spec.names, theta, strict=True))
    kc, tau = math.exp(p["log_kc"]), math.exp(p["log_tau"])
    amp = kc * inputs.carbs
    if spec.has("E1"):
        beta = math.exp(p["log_beta"])
        f1 = 1.0 / (1.0 + beta * inputs.other)
        amp = amp * f1
    if spec.has("E3"):
        f3 = np.maximum(1.0 + p["ks"] * inputs.debt, 0.1)
        amp = amp * f3
    U = S / tau
    with np.errstate(over="ignore"):
        phi = np.where(mask, U * np.exp(1.0 - U), 0.0)
    A = np.broadcast_to(amp[None, :], phi.shape)
    if spec.has("E2"):
        ka = math.exp(p["log_ka"])
        A = A * np.exp(-ka * E)
    C = A * phi
    g = p["b0"] + C.sum(axis=1)
    if not jac:
        return g
    cols = [np.ones(len(g)), C.sum(axis=1), (C * (U - 1.0)).sum(axis=1)]
    for ext in spec.extensions:
        if ext == "E1":
            cols.append((C * (-beta * inputs.other / (1.0 + beta * inputs.other))[None, :]).sum(axis=1))
        elif ext == "E2":
            cols.append((C * (-ka * E)).sum(axis=1))
        elif ext == "E3":
            cols.append((C * (inputs.debt / f3)[None, :]).sum(axis=1))
    return g, np.column_stack(cols)


def predict(
    theta: np.ndarray, spec: TwinSpec, inputs: TwinInputs, x: np.ndarray, until: np.ndarray | float | None = None
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    until = x if until is None else until
    S, mask, E = _design(inputs, x, until, spec.has("E2"))
    return _curve(theta, spec, inputs, S, mask, E)


def fit(
    inputs: TwinInputs,
    t_fit: np.ndarray,
    g_fit: np.ndarray,
    spec: TwinSpec,
    mu: np.ndarray,
    sd: np.ndarray,
    sigma: float = 20.0,
    x0: np.ndarray | None = None,
    max_nfev: int = 60,
) -> np.ndarray:
    """Penalised least squares: data residuals / sigma plus prior residuals / sd."""
    lo, hi = spec.bounds()
    if len(t_fit) == 0:
        return spec.clip(mu)
    t_fit = np.asarray(t_fit, dtype=float)
    g_fit = np.asarray(g_fit, dtype=float)
    S, mask, E = _design(inputs, t_fit, t_fit, spec.has("E2"))
    inv_sd = 1.0 / sd

    def fun(th):
        return np.concatenate([(g_fit - _curve(th, spec, inputs, S, mask, E)) / sigma, (th - mu) * inv_sd])

    def jac(th):
        _, J = _curve(th, spec, inputs, S, mask, E, jac=True)
        return np.vstack([-J / sigma, np.diag(inv_sd)])

    start = spec.clip(mu if x0 is None else x0)
    res = least_squares(fun, start, jac=jac, bounds=(lo, hi), method="trf", x_scale="jac", max_nfev=max_nfev)
    return res.x


def anchored(
    theta: np.ndarray,
    spec: TwinSpec,
    inputs: TwinInputs,
    now: np.ndarray,
    g_now: np.ndarray,
    horizons: np.ndarray,
    tau_r: float,
) -> np.ndarray:
    """Anchored forecasts, shape (len(now), len(horizons)). Meals after ``now`` are unknown."""
    now = np.atleast_1d(np.asarray(now, dtype=float))
    g_now = np.atleast_1d(np.asarray(g_now, dtype=float))
    h = np.asarray(horizons, dtype=float)
    x = (now[:, None] + h[None, :]).ravel()
    until = np.repeat(now, len(h))
    ghat = predict(theta, spec, inputs, x, until).reshape(len(now), len(h))
    ghat_now = predict(theta, spec, inputs, now, now)
    return ghat + (g_now - ghat_now)[:, None] * np.exp(-h / tau_r)[None, :]


# ----------------------------------------------------------------------------------- priors


@dataclass
class Prior:
    """P0 (population default) and P1 (profile-informed ridge prior), learned on training participants."""

    extensions: tuple[str, ...]
    p0_mu: list[float]
    p0_sd: list[float]
    medians: list[float] = field(default_factory=list)
    mean: list[float] = field(default_factory=list)
    scale: list[float] = field(default_factory=list)
    coef: list[list[float]] = field(default_factory=list)
    intercept: list[float] = field(default_factory=list)
    p1_sd: list[float] = field(default_factory=list)

    @property
    def spec(self) -> TwinSpec:
        return TwinSpec(tuple(self.extensions))

    def p0(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array(self.p0_mu), np.array(self.p0_sd)

    def p1(self, profile: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        if not self.coef:
            return self.p0()
        x = np.array([float(profile.get(f, np.nan)) for f in PROFILE_FEATURES])
        x = np.where(np.isnan(x), np.array(self.medians), x)
        z = (x - np.array(self.mean)) / np.array(self.scale)
        mu = np.array(self.intercept) + np.array(self.coef) @ z
        return self.spec.clip(mu), np.array(self.p1_sd)

    def to_dict(self) -> dict:
        return self.__dict__ | {"extensions": list(self.extensions)}

    @classmethod
    def from_dict(cls, d: dict) -> Prior:
        d = dict(d)
        d["extensions"] = tuple(d["extensions"])
        return cls(**d)


def fit_prior(thetas: dict[str, np.ndarray], profiles: pd.DataFrame, spec: TwinSpec) -> Prior:
    pids = sorted(thetas)
    theta = np.vstack([thetas[p] for p in pids])
    floor = np.array([SD_FLOOR[n] for n in spec.names])
    p0_mu = theta.mean(axis=0)
    p0_sd = np.maximum(theta.std(axis=0, ddof=1) if len(pids) > 1 else floor, floor)
    prior = Prior(tuple(spec.extensions), p0_mu.tolist(), p0_sd.tolist())
    if len(pids) < 4:
        return prior
    X = profiles.loc[pids, PROFILE_FEATURES].astype(float)
    medians = X.median().fillna(0.0)
    X = X.fillna(medians)
    mean, scale = X.mean(), X.std(ddof=0).replace(0, 1.0).fillna(1.0)
    Z = ((X - mean) / scale).to_numpy()
    ridge = Ridge(alpha=1.0).fit(Z, theta)
    resid = theta - ridge.predict(Z)
    prior.medians = medians.tolist()
    prior.mean = mean.tolist()
    prior.scale = scale.tolist()
    prior.coef = np.atleast_2d(ridge.coef_).tolist()
    prior.intercept = np.atleast_1d(ridge.intercept_).tolist()
    prior.p1_sd = np.maximum(resid.std(axis=0, ddof=1), floor).tolist()
    return prior
