"""Run the GlucoTwin pipeline for one track.

    python scripts/run_pipeline.py                 # evidence track (needs CGMacros in data/raw/cgmacros)
    python scripts/run_pipeline.py --demo          # synthetic track (no download)
    python scripts/run_pipeline.py --demo --smoke  # tiny synthetic run used by CI
    python scripts/run_pipeline.py --sensitivity   # also run the sensitivity analyses

Outputs: reports (evidence: reports/; synthetic: reports/synthetic/), the dashboard bundle
(models/<track>/) and processed tables (data/processed/<track>/, git-ignored). With
``--out DIR`` everything goes under DIR instead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from glucotwin.config import load_config  # noqa: E402
from glucotwin.pipeline import run  # noqa: E402
from glucotwin.schema import ValidationError  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true", help="run the synthetic track instead of the evidence track")
    ap.add_argument("--smoke", action="store_true", help="tiny synthetic cohort, fewer folds (CI)")
    ap.add_argument("--sensitivity", action="store_true", help="also run the sensitivity analyses")
    ap.add_argument("--out", default=None, help="write all outputs under this folder")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = ap.parse_args(argv)

    overrides = None
    if args.smoke:
        if not args.demo:
            ap.error("--smoke runs on the synthetic track; add --demo")
        sm = load_config(args.config)["synthetic"]["smoke"]
        overrides = {
            "cv": {"outer_folds": sm["outer_folds"], "inner_folds": sm["inner_folds"], "bootstrap": sm["bootstrap"]},
            "twin": {"identifiability_bootstraps": 3},
        }
    cfg = load_config(args.config, overrides)
    try:
        run("synthetic" if args.demo else "evidence", cfg, out=args.out, smoke=args.smoke, sensitivity=args.sensitivity)
    except ValidationError as exc:
        print(
            f"\nSTOPPED by a validation rule: {exc}\nNothing downstream was run. See APPROACH.md §2.1.", file=sys.stderr
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
