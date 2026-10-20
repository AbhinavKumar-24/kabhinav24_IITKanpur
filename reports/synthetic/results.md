# Results — synthetic track

> **SYNTHETIC DEMO.** Every value here comes from simulated data generated with a meal-response model like the twin itself. These numbers show the pipeline runs end to end; they are **not** evidence of accuracy.

Nested participant-wise cross-validation: 5 outer folds (reporting only), 4 inner folds (all selection). 95% intervals from 1000 bootstrap resamples of participants. 60 participants, 308 confirmed events, 216825 evaluable and 5474 unscorable windows.

## Event prediction (outer folds)

Events warned early = an alert 30–120 minutes before onset. Lead time is measured to the onset. Alert budget 3 per patient-day, 90-minute cooldown.

| Model | Warned early | Warned late | Missed | Median lead (min) | False alerts / patient-day | AUPRC | AUROC | Events | Alerts (unscorable) |
|---|---|---|---|---|---|---|---|---|---|
| M0 Trend baseline | 42% (30%–56%) | 52% (38%–65%) | 6% (1%–12%) | 25 (20–35) | 2.58 (2.37–2.78) | 0.11 (0.08–0.14) | 0.75 (0.69–0.81) | 308 | 2472 (51) |
| M1 Dynamic (LightGBM) | 89% (79%–98%) | 7% (1%–16%) | 4% (0%–10%) | 70 (65–80) | 2.95 (2.37–3.59) | 0.48 (0.31–0.61) | 0.95 (0.93–0.97) | 308 | 2841 (61) |
| M2 Fused (LightGBM) | 81% (66%–94%) | 14% (3%–26%) | 6% (1%–12%) | 70 (60–70) | 3.11 (2.23–4.03) | 0.40 (0.21–0.58) | 0.94 (0.91–0.97) | 308 | 2978 (66) |
| M3 Twin-enhanced (LightGBM) | 79% (65%–92%) | 16% (4%–29%) | 5% (1%–11%) | 70 (60–75) | 3.12 (2.26–4.01) | 0.44 (0.26–0.63) | 0.94 (0.90–0.97) | 308 | 2981 (61) |

**Selection inside each outer fold (inner cross-validation):**

| Fold | Dashboard model | Twin extensions |
|---|---|---|
| 1 | M3 | none |
| 2 | M1 | none |
| 3 | M1 | none |
| 4 | M1 | none |
| 5 | M1 | none |

## Twin extensions (all reported, selected or not)

Change in 60-minute forecast RMSE (extension minus base), twin fitted on each held-out participant's first 72 hours.

| Extension | Selected in folds | RMSE change, mg/dL (95% CI) | Participants |
|---|---|---|---|
| E1 protein/fat/fibre damping | 0 of 5 | 0.07 (0.05–0.09) | 60 |
| E2 observed post-meal activity | 0 of 5 | 0.09 (0.05–0.13) | 60 |
| E3 estimated sleep | 0 of 5 | 0.01 (-0.00–0.03) | 60 |

## Personalisation (cold start)

Twin 60-minute forecast RMSE (mg/dL) on each held-out participant's data from hour 72 on. The 0 h column is the prior alone; later columns add the participant's own first hours.

| Starting point | 0 h | 6 h | 24 h | 72 h |
|---|---|---|---|---|
| Population default (P0) | 18.5 (16.7–20.7) | 17.1 (15.8–18.6) | 16.7 (15.5–18.1) | 16.6 (15.4–18.0) |
| Profile-informed prior (P1 → P2) | 16.8 (15.6–18.2) | 16.8 (15.6–18.3) | 16.7 (15.4–18.1) | 16.6 (15.4–18.0) |

## Glucose forecast (secondary)

| Method | Horizon | RMSE mg/dL (95% CI) | MAE mg/dL (95% CI) | Clarke A+B | Points |
|---|---|---|---|---|---|
| Personal twin (anchored) | +30 min | 10.6 (9.9–11.2) | 6.9 (6.7–7.1) | 99% | 73918 |
| Personal twin (anchored) | +60 min | 17.1 (15.8–18.8) | 10.2 (9.7–10.8) | 99% | 73460 |
| Trend baseline | +30 min | 17.2 (16.6–17.9) | 12.6 (12.3–12.9) | 99% | 73918 |
| Trend baseline | +60 min | 32.4 (31.0–34.0) | 23.4 (22.6–24.3) | 97% | 73460 |

## Baseline propensity (profile only)

A lab profile cannot time an excursion; this relates HbA1c to each participant's overall excursion rate.

Spearman correlation, HbA1c vs confirmed events per day: 0.53 (0.28–0.73) (60 participants).

## Twin parameter spread (identifiability check)

Median standard deviation of each parameter when the base twin is refitted on bootstrap resamples of a participant's days. Parameters are descriptive, not physiological.

| Parameter | Median SD | Participants |
|---|---|---|
| baseline_mgdl | 1.009 | 30 |
| rise_per_g_carb | 0.025 | 30 |
| time_to_peak_min | 1.401 | 30 |

## By diagnosis group (descriptive only)

| Model | Group | Warned early | False alerts / patient-day | Participants |
|---|---|---|---|---|
| M0 | healthy | 50% | 1.95 | 12 |
| M0 | prediabetes | 42% | 2.63 | 24 |
| M0 | t2d | 42% | 2.85 | 24 |
| M1 | healthy | 100% | 2.43 | 12 |
| M1 | prediabetes | 67% | 1.91 | 24 |
| M1 | t2d | 92% | 4.26 | 24 |
| M2 | healthy | 75% | 1.62 | 12 |
| M2 | prediabetes | 42% | 1.64 | 24 |
| M2 | t2d | 88% | 5.33 | 24 |
| M3 | healthy | 25% | 1.77 | 12 |
| M3 | prediabetes | 58% | 1.62 | 24 |
| M3 | t2d | 89% | 5.29 | 24 |

Sensitivity analyses were not run (use `--sensitivity`).
