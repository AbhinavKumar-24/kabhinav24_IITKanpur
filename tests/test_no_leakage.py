"""Deleting all data after t leaves every feature at t unchanged (APPROACH.md §3, §5)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from glucotwin import twin as tw
from glucotwin.features import TWIN_FEATURES, daily_params, prepare_participant, twin_features


def test_features_at_t_ignore_the_future(evidence_td, pdatas, evidence_cfg):
    pid = sorted(pdatas)[2]
    full = pdatas[pid]
    spec = tw.TwinSpec(("E2",))  # include the activity extension, the easiest place to leak
    mu, sd = spec.generic_prior()
    full_params = daily_params(full, spec, mu, sd, evidence_cfg)
    full_tf = twin_features(full, spec, full_params, 45.0)
    m = full.moments
    for frac in (0.35, 0.6, 0.85):
        k = int(frac * (len(m) - 1))
        when = pd.Timestamp(m["time"].iloc[k])
        cut = prepare_participant(evidence_td.truncated(pid, when), pid, evidence_cfg)
        row = cut.moments[cut.moments["time"] == when]
        assert len(row) == 1, "the moment must still be eligible with only past data"
        feats = [c for c in m.columns if c not in ("status", "y", "row")]
        a = m.iloc[k][feats].drop(["time"]).astype(float).to_numpy()
        b = row.iloc[0][feats].drop(["time"]).astype(float).to_numpy()
        assert np.allclose(a, b, equal_nan=True)
        cut_params = daily_params(cut, spec, mu, sd, evidence_cfg)
        cut_tf = twin_features(cut, spec, cut_params, 45.0)
        assert np.allclose(
            full_tf.iloc[k][TWIN_FEATURES].astype(float),
            cut_tf.loc[row.index[0], TWIN_FEATURES].astype(float),
            equal_nan=True,
        )
