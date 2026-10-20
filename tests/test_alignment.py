"""Interval-aware coverage: routine sampling is never counted as missing (APPROACH.md §2.5)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from glucotwin.align import build_timeline, wear_coverage, window_coverage


def _readings(minutes: np.ndarray) -> pd.DataFrame:
    t0 = pd.Timestamp("2024-01-01 06:00")
    return pd.DataFrame(
        {
            "time": t0 + pd.to_timedelta(minutes, unit="m"),
            "glucose": 100 + np.arange(len(minutes), dtype=float),
            "censored": np.zeros(len(minutes), dtype=np.int8),
        }
    )


def test_regular_libre_and_dexcom_have_full_coverage():
    libre = np.arange(0, 600, 15, dtype=float)
    dexcom = np.arange(0, 600, 5, dtype=float)
    assert wear_coverage(_readings(libre)["time"], 15) == 1.0
    assert wear_coverage(_readings(dexcom)["time"], 5) == 1.0
    q = np.array([300.0, 450.0])
    assert np.allclose(window_coverage(libre, q - 60, q, 15), 1.0)
    assert np.allclose(window_coverage(dexcom, q - 60, q, 5), 1.0)
    tl = build_timeline(_readings(libre), 15, 20, pd.Timestamp("2024-01-01 06:00"))
    assert (tl["src"] == "imputed").sum() == 0  # 15-minute spacing is routine, not a gap


def test_short_gap_imputed_long_gap_left_empty():
    t0 = pd.Timestamp("2024-01-01 06:00")
    with_short_gap = np.r_[np.arange(0, 60, 5), np.arange(65, 120, 5)]  # the reading at 60 is missing
    tl = build_timeline(_readings(with_short_gap.astype(float)), 5, 20, t0)
    imputed = tl[tl["src"] == "imputed"]
    assert len(imputed) == 1 and np.isclose(imputed["t"].iloc[0], 60.0)
    assert np.isclose(window_coverage(with_short_gap.astype(float), np.array([60.0]), np.array([120.0]), 5)[0], 11 / 12)
    long_gap = np.r_[np.arange(0, 60, 5), np.arange(85, 120, 5)]  # 30 minutes between readings
    tl2 = build_timeline(_readings(long_gap.astype(float)), 5, 20, t0)
    assert (tl2["src"] == "imputed").sum() == 0
