"""The Streamlit dashboard runs end to end on the committed synthetic bundle (all five tabs)."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "models" / "synthetic" / "bundle.json"


@pytest.mark.skipif(not BUNDLE.exists(), reason="run `python scripts/run_pipeline.py --demo` first")
def test_dashboard_renders_without_errors():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app" / "dashboard.py"), default_timeout=900).run()
    assert not at.exception, at.exception
    assert any("SYNTHETIC DEMO" in w.value for w in at.warning)
    assert len(at.tabs) == 5
    assert any(
        "not a physiological simulation" in i.value.lower() or "simulated estimate" in i.value.lower() for i in at.info
    )
