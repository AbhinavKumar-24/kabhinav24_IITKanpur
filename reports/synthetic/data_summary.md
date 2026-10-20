# Data summary — synthetic track

> **SYNTHETIC DEMO.** Every value here comes from simulated data generated with a meal-response model like the twin itself. These numbers show the pipeline runs end to end; they are **not** evidence of accuracy.

Generated 2026-10-20 09:25 UTC by `scripts/run_pipeline.py`.

## Validation checks

| Check | Status | Detail |
|---|---|---|
| Synthetic track | info | Every value is synthetic: Synthea EHRs plus simulated streams. Demo only, never evidence. |

## Counts

| Item | Value |
|---|---|
| Participants in analysis / total | 60 / 60 |
| Groups | {'t2d': 24, 'prediabetes': 24, 'healthy': 12} |
| Days of CGM | 834.7 |
| Mean primary-sensor coverage (observed / expected readings) | 99% |
| Imputed share of timeline rows | 0% |
| Censored readings (<= 40 / >= 400) | 11 / 0 |
| Eligible prediction moments | 222299 |
| Evaluable / unscorable windows | 216825 / 5474 |
| Confirmed events (total; median, min–max per participant) | 308; 0, 0–58 |
