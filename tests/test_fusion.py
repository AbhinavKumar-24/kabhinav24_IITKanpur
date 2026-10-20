"""The two tracks never cross; each fuses its own static and dynamic streams by its own ID."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from glucotwin.ingest_fhir import parse_bundle
from glucotwin.simulate import simulate_track

ROOT = Path(__file__).resolve().parents[1]
SYN = ROOT / "data" / "synthetic"


def test_evidence_participants_join_labs_and_sensors(evidence_td):
    for pid in evidence_td.profiles.index:
        assert pid.startswith("CGM-")
        assert (evidence_td.cgm["pid"] == pid).any()
        assert (evidence_td.meals["pid"] == pid).any()
    assert not evidence_td.cgm["pid"].str.startswith("SYN-").any()


def test_synthetic_profiles_and_ids_are_separate():
    prof = pd.read_csv(SYN / "synthetic_profiles.csv")
    assert prof["pid"].str.startswith("SYN-").all() and prof["pid"].is_unique
    assert prof["group"].value_counts().to_dict() == {"t2d": 80, "prediabetes": 80, "healthy": 40}
    assert {"prov_hba1c_pct", "prov_fasting_insulin"} <= set(prof.columns)


def test_fhir_sample_ingests_without_identifiers():
    for path in sorted((SYN / "fhir_sample").glob("*.json")):
        text = path.read_text(encoding="utf-8")
        assert '"name"' not in text and '"address"' not in text
        rec = parse_bundle(path, date(2026, 10, 1))
        assert rec is not None and rec["group"] in {"healthy", "prediabetes", "t2d"}
        assert path.stem.endswith(rec["group"])


def test_synthetic_track_is_deterministic(cfg):
    prof = pd.read_csv(SYN / "synthetic_profiles.csv").head(3)
    a = simulate_track(prof, cfg, days=2)
    b = simulate_track(prof, cfg, days=2)
    pd.testing.assert_frame_equal(a.cgm, b.cgm)
    assert set(a.cgm["pid"]) == set(prof["pid"])
    assert a.track == "synthetic"
