# Model card — GlucoTwin

## Intended use

- **Purpose:** a clinician-facing research prototype. It estimates the risk that glucose rises
  above 180 mg/dL and stays there for at least 15 minutes, starting within the next 2 hours.
  It shows the forecast, the main factors and the data quality behind it.
- **Not for:** diagnosis, treatment or dosing decisions, emergencies, or use without clinician
  oversight.
- **Status:** not a medical device. Not validated in Indian patients.

## Models

| Model | What it is |
|---|---|
| M0 | Trend baseline: projected glucose at +60 minutes from the last 30 minutes' slope |
| M1 | LightGBM on 16 dynamic features (glucose trend, meals, activity, heart rate, time of day) |
| M2 | M1 + 5 lab-profile features (HbA1c, fasting glucose, HOMA-IR, BMI, age) |
| M3 | M2 + 4 personal-twin features |

- **LightGBM settings** are fixed in `config.yaml` (300 trees, 15 leaves, balanced class
  weights). Scores are uncalibrated.
- **Dashboard model:** chosen by inner cross-validation (best early-warning rate at the same alert
  budget). The choice is recorded in `models/<track>/bundle.json`.

**Personal twin:** glucose = baseline + Σ meals (rise per gram of carbohydrate × response curve
peaking at a personal time to peak).

- **Fitting:** penalised least squares on observed, non-imputed, non-censored readings.
- **Priors:** a population default (P0) or a lab-informed ridge prior (P1).
- **Updates:** refitted once per day on data before midnight.
- **Extensions:** E1 protein/fat/fibre damping, E2 observed post-meal activity, E3 estimated sleep.
  Each is used only if inner cross-validation shows lower forecast error.
- **Nature:** phenomenological. Its parameters are descriptive, not physiological.

**Explanations:** TreeSHAP contributions computed by LightGBM (`pred_contrib`), summed by
feature family. They are associations, not causes.

## Training and evaluation data

| Track | Data | Use |
|---|---|---|
| Evidence | CGMacros v1.0.0: 45 US adults, 10 days, Dexcom primary sensor | All reported metrics |
| Synthetic | Synthea v3.4.0 EHRs + simulated streams with Indian meal patterns | Demo and tests only; never evidence |

## Evaluation protocol

- **Cross-validation:** nested and participant-wise (5 outer folds for reporting, 4 inner folds
  for every selection).
- **Uncertainty:** bootstrap over participants.
- **Event-level metrics:** events warned early (alert 30–120 minutes before onset), late and
  missed; median lead time; false alerts per patient-day; unscorable alerts.
- **Secondary metrics:** row-level AUPRC and AUROC; twin vs trend forecast RMSE, MAE and Clarke
  A+B at +30 and +60 minutes; a personalisation (cold-start) experiment.
- **Outputs:** `scripts/run_pipeline.py` writes `reports/data_summary.md`, `reports/results.md`
  and `reports/metrics.json`. They are never edited by hand.

## Results

Not yet evaluated on CGMacros — the evidence-track run happens locally after the dataset is
downloaded. The synthetic demo results in `reports/synthetic/` only show that the pipeline runs;
they are not evidence of accuracy.

## Limitations and risks

- 45 participants and 10 days each give wide uncertainty; results are exploratory feasibility
  evidence.
- Participants are US adults on US-style meals. Nothing here establishes performance for Indian
  patients or Indian meals.
- Meal macros are logged estimates; sleep on the evidence track is a heuristic estimate.
- Real low-glucose events are too rare to evaluate, so low alerts are out of scope.
- Rows from the same person are correlated: LightGBM models may not beat the trend baseline. If
  so, that is reported as found.
- What-if scenarios are simulated estimates from the twin, not causal effects or
  recommendations.

## Ethical and licence notes

- No identifiable data is used. Synthea identifiers are never read into profiles.
- CGMacros is CC BY-NC-SA 4.0: used non-commercially, with attribution. Processed tables and
  models trained on it are not redistributed.
- A real-world deployment would need clinical validation, regulatory review as software as a
  medical device (CDSCO) and DPDP Act compliance.
