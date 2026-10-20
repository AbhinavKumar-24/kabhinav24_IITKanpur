"""Shared fixtures. Every test runs on small synthetic files; the real CGMacros data is never needed."""

from __future__ import annotations

from pathlib import Path

import pytest

from glucotwin.config import deep_update, load_config
from glucotwin.ingest_cgmacros import ingest
from glucotwin.simulate import write_cgmacros_like

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def cfg() -> dict:
    return load_config(ROOT / "config.yaml")


@pytest.fixture(scope="session")
def cgmacros_like(tmp_path_factory, cfg):
    """Nine participants x 3 days shaped like CGMacros (Dexcom blank rows, Libre interpolated)."""
    folder = tmp_path_factory.mktemp("cgmacros_like")
    counts = write_cgmacros_like(
        folder, cfg, n_participants=9, days=3, seed=5, dexcom_mode="blank", libre_mode="interpolated"
    )
    return folder, counts


@pytest.fixture(scope="session")
def evidence_cfg(cfg, cgmacros_like) -> dict:
    folder, counts = cgmacros_like
    return deep_update(
        cfg,
        {
            "paths": {"raw_cgmacros": str(folder)},
            "cgmacros": {"expected_group_counts": counts},
            "cv": {"outer_folds": 3, "inner_folds": 2, "bootstrap": 50},
            "twin": {"identifiability_bootstraps": 2},
        },
    )


@pytest.fixture(scope="session")
def evidence_td(evidence_cfg, cgmacros_like):
    return ingest(cgmacros_like[0], evidence_cfg)


@pytest.fixture(scope="session")
def pdatas(evidence_td, evidence_cfg):
    from glucotwin.features import prepare_track

    return prepare_track(evidence_td, evidence_cfg)


@pytest.fixture(scope="session")
def bundle(evidence_td, pdatas, evidence_cfg):
    from glucotwin.pipeline import build_bundle

    return build_bundle(evidence_td, pdatas, evidence_cfg, clock_ok=True)
