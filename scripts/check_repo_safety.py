"""Repository safety check, run in CI and before every commit.

    python scripts/check_repo_safety.py

Fails if the repository would publish raw or processed data, CGMacros files, participant-level
tables, models trained on the evidence track, pickles, files over 5 MB, or credential-like
strings. In a Git checkout it checks tracked and untracked-but-not-ignored files; outside Git it
walks the folder, skipping the git-ignored data folders.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 5 * 1024 * 1024
ALLOWED_REPORTS = {"data_summary.md", "results.md", "metrics.json", "readme.md", ".gitkeep"}
DATA_FILE = re.compile(r"(cgmacros-?\d+\.csv|bio\.csv|gut_health_test\.csv|microbe?s\.csv)$")
SECRETS = [
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN (RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{36}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{40,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"sk-(ant-)?[A-Za-z0-9_-]{32,}"),
]
TEXT_SUFFIXES = {".py", ".md", ".txt", ".yaml", ".yml", ".json", ".toml", ".cfg", ".sh", ".csv", ".ini", ".env"}
SKIP_TREE = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "ci-smoke",
    "ci-full-demo",
    "test-results",
}
SKIP_TREE_PATHS = ("data/raw/", "data/processed/", "data/synthetic/synthea_raw/", "models/evidence/")


def candidate_files(root: Path = ROOT) -> list[str]:
    if (root / ".git").exists():
        out = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return sorted({p for p in out.split("\0") if p})
    files = []
    for path in root.rglob("*"):
        rel = path.relative_to(root).as_posix()
        if (
            path.is_dir()
            or any(part in SKIP_TREE for part in path.parts)
            or rel.startswith(SKIP_TREE_PATHS)
            or "egg-info" in rel
        ):
            continue
        files.append(rel)
    return sorted(files)


def problems(files: list[str], root: Path = ROOT) -> list[str]:
    issues = []
    for rel in files:
        path = root / rel
        low = rel.lower()
        name = Path(low).name
        if low.startswith(("data/raw/", "data/processed/")) and name not in {"readme.md", ".gitkeep"}:
            issues.append(f"{rel}: raw or processed data must never be committed")
        if DATA_FILE.search(name) or (name.endswith(".zip") and "cgmacros" in name):
            issues.append(f"{rel}: looks like a CGMacros data file")
        if low.startswith("reports/") and not low.startswith("reports/synthetic/") and name not in ALLOWED_REPORTS:
            issues.append(f"{rel}: only aggregate reports may be committed under reports/")
        if (
            low.startswith("models/")
            and not low.startswith("models/synthetic/")
            and name not in {"readme.md", ".gitkeep"}
        ):
            issues.append(f"{rel}: only synthetic-track demo models may be committed")
        if path.suffix.lower() in {".pkl", ".pickle", ".joblib"}:
            issues.append(f"{rel}: pickled objects are not allowed")
        if not path.is_file():
            continue
        size = path.stat().st_size
        if size > MAX_BYTES:
            issues.append(f"{rel}: {size / 1e6:.1f} MB is over the 5 MB limit")
        if path.suffix.lower() in TEXT_SUFFIXES and size < 2_000_000:
            text = path.read_text(encoding="utf-8", errors="ignore")
            for pattern in SECRETS:
                if pattern.search(text):
                    issues.append(f"{rel}: contains a credential-like string ({pattern.pattern[:12]}...)")
                    break
    return issues


def main() -> int:
    files = candidate_files()
    issues = problems(files)
    if issues:
        print("Repository safety check FAILED:")
        for issue in issues:
            print(f"  - {issue}")
        return 1
    print(f"Repository safety check passed ({len(files)} files checked).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
