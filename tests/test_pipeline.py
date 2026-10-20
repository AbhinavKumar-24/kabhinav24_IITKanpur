"""End-to-end run of the evidence path on CGMacros-shaped fixtures, plus what-if and reasons."""

from __future__ import annotations

import json

import numpy as np

from glucotwin import twin as tw
from glucotwin.explain import reasons
from glucotwin.models import MODELS
from glucotwin.pipeline import load_bundle, run
from glucotwin.replay import precompute
from glucotwin.whatif import DISCLAIMER, scenario


def test_evidence_pipeline_end_to_end(tmp_path, evidence_cfg):
    out = run("evidence", evidence_cfg, out=str(tmp_path), log=lambda *a: None)
    rep = tmp_path / "reports" / "evidence"
    for name in ("data_summary.md", "results.md", "metrics.json"):
        assert (rep / name).exists()
    metrics = json.loads((rep / "metrics.json").read_text())
    assert set(metrics["results"]["models"]) == set(MODELS)  # every model reported
    assert len(metrics["results"]["selection"]) == evidence_cfg["cv"]["outer_folds"]
    bundle, fitted = load_bundle(tmp_path / "models" / "evidence")
    assert bundle["selected_model"] in MODELS and set(fitted) == set(MODELS)
    assert out["counts"]["unscorable_windows"] >= 0


def test_whatif_and_reasons(pdatas, bundle, evidence_cfg):
    bdict, fitted = bundle
    p = pdatas[sorted(pdatas)[-1]]
    pre = precompute(p, bdict, fitted, evidence_cfg)
    k = len(p.moments) // 2
    now, g_now = float(p.moments["t"].iloc[k]), float(p.moments["g_now"].iloc[k])
    day = int(p.moments["day_idx"].iloc[k])
    spec = tw.TwinSpec(tuple(bdict["spec"]))
    res = scenario(
        p, spec, pre.params[day], bdict["tau_r"], bdict["band"], now, g_now, "next", 60.0, 15.0, evidence_cfg
    )
    assert len(res["baseline"]) == len(res["scenario"]) == len(res["horizons"])
    assert np.isclose(res["baseline"][0], g_now) and res["disclaimer"] == DISCLAIMER
    assert (
        res["scenario_summary"]["peak"] >= res["baseline_summary"]["peak"] - 1e-6
    )  # adding carbs cannot lower the curve
    why = reasons(pre.model, pre.X.iloc[[k]])
    assert why and all(r["text"] for r in why)
