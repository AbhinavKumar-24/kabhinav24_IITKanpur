"""The repository safety check catches data, participant-level tables, evidence models and secrets."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_repo_safety", ROOT / "scripts" / "check_repo_safety.py")
safety = importlib.util.module_from_spec(spec)
spec.loader.exec_module(safety)


def test_flags_unsafe_files(tmp_path):
    bad = {
        "data/raw/cgmacros/bio.csv": "subject,Age\n1,30\n",
        "reports/participant_rows.csv": "pid,glucose\n",
        "models/evidence/model_M2.txt": "tree\n",
        "notes.md": "token " + "ghp_" + "a" * 36 + "\n",
    }
    for rel, text in bad.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    issues = safety.problems(sorted(bad), tmp_path)
    assert len(issues) >= 4
    assert any("credential" in i for i in issues) and any("CGMacros" in i for i in issues)


def test_allows_synthetic_outputs_and_aggregate_reports(tmp_path):
    ok = [
        "reports/results.md",
        "reports/synthetic/results.md",
        "models/synthetic/bundle.json",
        "src/glucotwin/ingest_cgmacros.py",
    ]
    for rel in ok:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ok\n")
    assert safety.problems(ok, tmp_path) == []
