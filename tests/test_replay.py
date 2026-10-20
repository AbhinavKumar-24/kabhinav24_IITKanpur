"""Replay at time t never uses data after t (APPROACH.md §10)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from glucotwin.features import prepare_participant
from glucotwin.replay import precompute, replay_state, state_at


def test_replay_matches_truncated_recomputation(evidence_td, pdatas, bundle, evidence_cfg):
    bdict, fitted = bundle
    pid = sorted(pdatas)[-1]
    full = prepare_participant(evidence_td, pid, evidence_cfg)
    pre = precompute(full, bdict, fitted, evidence_cfg)
    times = full.moments["time"].iloc[[50, len(full.moments) // 2, len(full.moments) - 40]]
    for when in times:
        when = pd.Timestamp(when)
        now = float((when - full.t0) / pd.Timedelta(minutes=1))
        fast = state_at(pre, now, bdict, evidence_cfg)
        slow = replay_state(evidence_td, pid, when, bdict, fitted, evidence_cfg)
        assert fast["status"] == slow["status"]
        assert np.isclose(fast["g_now"], slow["g_now"])
        assert np.isclose(fast["risk"], slow["risk"], equal_nan=True)
        assert np.allclose(fast["alerts_so_far"], slow["alerts_so_far"])
        assert np.allclose(fast["forecast"], slow["forecast"])
        assert len(fast["events"]) == len(slow["events"])
        assert fast["trace"]["t"].max() <= now
