"""Stop rules and data checks (APPROACH.md §2.1)."""

from __future__ import annotations

import shutil

import numpy as np
import pandas as pd
import pytest

from glucotwin.config import deep_update
from glucotwin.ingest_cgmacros import ingest
from glucotwin.schema import ValidationError
from glucotwin.simulate import write_cgmacros_like
from glucotwin.validate import censor_flags, hba1c_groups, reconstruct_native, require_increasing


@pytest.mark.parametrize("mode", ["carried", "interpolated"])
def test_native_readings_identical_across_row_layouts(tmp_path, cfg, mode):
    """Blank, carried-forward and interpolated 1-minute rows give the same native readings."""
    ref_dir, other_dir = tmp_path / "blank", tmp_path / mode
    counts = write_cgmacros_like(
        ref_dir, cfg, n_participants=3, days=2, seed=9, dexcom_mode="blank", libre_mode="blank"
    )
    write_cgmacros_like(other_dir, cfg, n_participants=3, days=2, seed=9, dexcom_mode=mode, libre_mode=mode)
    c = deep_update(cfg, {"cgmacros": {"expected_group_counts": counts}})
    a, b = ingest(ref_dir, c), ingest(other_dir, c)
    cols = ["pid", "time", "glucose", "sensor"]
    pd.testing.assert_frame_equal(
        a.cgm[cols].sort_values(cols).reset_index(drop=True),
        b.cgm[cols].sort_values(cols).reset_index(drop=True),
    )
    assert set(b.meta["sampling_modes"]["CGM-001"].values()) == {
        "carried_forward" if mode == "carried" else "interpolated"
    }


def test_unidentifiable_sampling_is_a_stop(cfg):
    rng = np.random.default_rng(0)
    times = pd.Series(pd.date_range("2024-01-01", periods=600, freq="min"))
    values = pd.Series(100 + rng.normal(0, 10, 600))  # every minute different, not linear
    _, mode = reconstruct_native(times, values, 5)
    assert mode == "unknown"


def test_hba1c_percent_check():
    vals = pd.Series([5.2, 5.5, 6.0, 6.3, 7.1, 8.0])
    groups, detail = hba1c_groups(vals, (5.7, 6.4), {"healthy": 2, "prediabetes": 2, "t2d": 2})
    assert list(groups) == ["healthy", "healthy", "prediabetes", "prediabetes", "t2d", "t2d"]
    assert "mmol/mol" in detail
    with pytest.raises(ValidationError):
        hba1c_groups(vals, (5.7, 6.4), {"healthy": 3, "prediabetes": 2, "t2d": 1})
    with pytest.raises(ValidationError):
        hba1c_groups(pd.Series([38.0, 42.0, 55.0]), (5.7, 6.4), None)  # values in mmol/mol


def test_lab_error_codes_mets_and_censoring(evidence_td):
    prof = evidence_td.profiles
    assert np.isnan(prof.loc["CGM-003", "ldl"])  # fixture sets LDL = 800 here
    assert np.isnan(prof.loc["CGM-005", "vldl"])  # and VLDL = 400 here
    assert evidence_td.activity["mets"].max() < 25
    assert list(censor_flags(np.array([40.0, 100.0, 400.0]), 40, 400)) == [-1, 0, 1]
    assert {c.name for c in evidence_td.checks} >= {
        "HbA1c units (stop rule)",
        "Join: bio.csv rows to participant files",
    }


def test_join_failure_is_a_stop(tmp_path, cgmacros_like, evidence_cfg):
    folder = tmp_path / "copy"
    shutil.copytree(cgmacros_like[0], folder)
    shutil.rmtree(folder / "CGMacros-001")
    with pytest.raises(ValidationError, match="Join check failed"):
        ingest(folder, evidence_cfg)


def test_timestamps_must_increase():
    with pytest.raises(ValidationError):
        require_increasing(pd.Series(pd.to_datetime(["2024-01-01 00:02", "2024-01-01 00:01"])), "x.csv")
