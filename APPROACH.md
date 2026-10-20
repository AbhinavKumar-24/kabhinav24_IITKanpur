# GlucoTwin — Approach

Version 3.2 · 20 October 2026 · implemented as described here · revision notes at the end

This document covers how GlucoTwin is built and evaluated: data, prediction task, personal twin, event model, evaluation, alerts, dashboard and milestones.

---

## 0. Scope

**Keep**

- A 2-hour excursion prediction as the primary task
- A personal glucose-response twin
- Participant-wise evaluation against simple baselines
- A Streamlit dashboard with historical replay
- Provenance tags and synthetic-data badges throughout
- A small, transparent what-if demonstration

**Simplified**

- Twin: 3 parameters to start
- Features: about 25, fixed before evaluation
- Models: M0 → M3, all reported
- Twin updates: once per replay day
- Alerts: tuned only after the event definition and evaluation are fixed

**Postponed — only if time remains**

- Synthetic sleep-stage and HRV effects in any predictive model
- The genetic marker as a visible dashboard feature
- Low-glucose alerts (too few real events to evaluate)
- A censored-data loss for the twin (only needed if censoring turns out to be common)
- Any optimisation of meals or activity, and any causal claims
- Live clinical integration, and claims about performance in Indian patients

---

## 1. Design principles

1. **Real evidence, separate synthetic demo.** Metrics come only from real, anonymised data. Synthetic data demonstrates the brief's simulated-EHR pathway and Indian scenarios, and is never joined to a real participant.
2. **Every value carries its origin.** Each value is tagged observed, derived, imputed or synthetic, from ingestion through to the dashboard.
3. **Live rules use only the past.** Rules that make a prediction or fire an alert use only data at or before *t*. Rules that score an outcome may look ahead, but only in offline evaluation.
4. **Report everything, select inside.** Every model and extension is reported. Any choice of what to keep is made by inner cross-validation, never by the reported results.
5. **Claims proportional to evidence.** This is an exploratory feasibility study on a small public cohort, not a clinically validated predictor.

---

## 2. Data

### 2.1 Evidence track — CGMacros (real, anonymised)

**Facts from the dataset's PhysioNet page** (checked 20 Oct 2026):

| Item | Source description |
|---|---|
| Participants | 45 — 15 healthy, 16 prediabetes, 14 Type 2 diabetes — aged 18–69, recruited 2021–2024 at the Sansum Diabetes Research Institute, Santa Barbara. Grouped by HbA1c: <5.7%, 5.7–6.4%, >6.4%. People with Type 2 diabetes on medicines other than metformin were excluded. |
| Sensors | Abbott FreeStyle Libre Pro (**15-minute** sampling, upper arm) and Dexcom G6 Pro (**5-minute** sampling, abdomen). Both were blinded. A Fitbit Sense was used to log exercise. |
| Meals | 10 days. Breakfast: protein shakes with varying macronutrients. Lunch: a fast-casual restaurant chain. Dinner: the participant's own choice. At least 3 hours between meals. Meal times come from before/after photos. Logged values: carbs, protein, fat, fibre, calories and the proportion eaten. |
| Files | One CSV per participant (`CGMacros-0XX.csv`) with rows at **1-minute** intervals. `bio.csv` holds demographics, anthropometrics and labs. Microbiome files are not used. |
| Dates | Shifted per participant by ±N whole days (365 < N < 720) |
| Licence | CC BY-NC-SA 4.0, open access |

**Validation.** If a stop rule fails, the pipeline stops with a clear message, and `reports/data_summary.md` records the result of every check.

| Check | Rule | If it fails |
|---|---|---|
| **Native sampling** | Rows are 1-minute, but Dexcom samples every 5 minutes and Libre every 15. Determine whether the rows between native readings are blank, carried forward or interpolated. Reconstruct each sensor's native readings; in-between values are never treated as observed. | Stop if native readings cannot be identified |
| **HbA1c units** | The dictionary labels the column mmol/mol, but values run 4.6–8.5 and the dataset's group thresholds (5.7%, 6.4%) are in percent. Treat the values as percent only if groups recomputed under that reading reproduce 15 / 16 / 14. The report records the conflict, the evidence and the check. | Stop |
| Lab error codes | LDL = 800 and VLDL = 400 are calculation errors | Set to missing |
| METs scale | Stored × 10; divide by 10, then check a plausible range | Stop if out of range |
| **Censored glucose** | Sensors report 40–400. Keep the raw value; flag 40 as "≤ 40" and 400 as "≥ 400"; count how often each occurs. | Flag and count |
| **Primary sensor** | Dexcom (native 5-minute) is primary for every participant with ≥ 70% of expected Dexcom readings over their wear period. The Libre series is kept for cross-checks and a sensitivity analysis. | Participants below 70% are excluded from the primary analysis and counted |
| Sensor agreement | Report Dexcom–Libre agreement at matched times for each participant | Report only |
| Timestamps | Must increase within each participant. Dates are never compared across people. The shift is in whole days, so time of day should be intact; a meal-order check confirms this (breakfast before lunch before dinner on most days). | Stop if they don't increase |
| Meals | Use macros × proportion eaten, where the proportion is recorded | Report missing values |
| Join | Every `bio.csv` row matches exactly one participant file | Stop |

### 2.2 Static profile (evidence track)

- **Observed:** age, sex, BMI, HbA1c, fasting glucose, fasting insulin, triglycerides, HDL, total cholesterol.
- **Derived:**
  - HOMA-IR = fasting insulin (µU/mL) × fasting glucose (mg/dL) / 405
  - triglyceride/HDL ratio
  - diagnosis group

### 2.3 Sleep on the evidence track

- **Estimate.** Sleep is the longest low-activity, low-heart-rate block between 20:00 and 11:00. It is computed only on nights with at least 70% overnight wear coverage. The Fitbit was given to log exercise, so overnight wear may be sparse.
- **Labelling.** It is stored as *derived — heuristic*, with its coverage, and displayed as "estimated".
- **Use in models.** It stays out of the models unless extension E3 passes its test (§4.5).
- **Not available.** HRV and sleep stages are not in this dataset, so the evidence track does not use them.

### 2.4 Synthetic track (demo and brief-compliance only)

1. **Simulated EHRs.** Synthea v3.4.0 generates 1,000 adults aged 25–75 (seed 42, reference date 2026-10-01), exported as FHIR R4. The committed run produced 1,116 bundles: 976 living patients, 975 aged 25–75 (454 healthy, 432 prediabetes, 89 Type 2 diabetes by Synthea's conditions).
   - We parse conditions, medications and observations, and drop all identifiers.
   - We select 200 patients: 80 Type 2 diabetes, 80 prediabetes, 40 normoglycaemic.
   - Selection is weighted toward an approximate Indian adult age mix set in `config.yaml`.
   - HbA1c and fasting glucose come from Synthea when present and plausible (HbA1c 4–14%, glucose 60–400 mg/dL); otherwise they are simulated from group ranges. Fasting insulin is always simulated, because Synthea does not model it. Every field's origin is in a `prov_*` column.
2. **Simulated streams for those synthetic patients:**
   - meals from an Indian meal library (poha, idli–sambar, rice–dal–sabzi, roti–sabzi, khichdi, chai with biscuits and others), with macros checked against the Indian Food Composition Tables 2017 (Longvah et al., ICMR-NIN) and stored in `config.yaml` with a source column
   - activity, sleep, heart rate and HRV
   - CGM generated by the twin at a 5-minute cadence, plus physiological noise, sensor noise and dropouts
3. **Genetic marker.** A simulated TCF7L2 risk-allele count is stored in the synthetic profile only.
4. **Linkage.** A synthetic EHR joins only to the streams of the same synthetic patient.
5. **Use.** Demo mode, the hosted demo, stress tests and Indian-scenario demonstrations. **Never metrics** — its glucose is generated by the twin, so evaluating on it would be circular. The demo evaluation runs on a stratified 60-patient subset; the dashboard shows all 200 patients.

### 2.5 Alignment, coverage and provenance

- **Timeline.** Each participant's timeline is anchored to the native Dexcom reading times (about every 5 minutes). A fixed clock grid could split or duplicate readings, so none is used.
- **Coverage is interval-aware:**
  - expected readings in a window = window length ÷ native interval
  - coverage = native readings present ÷ expected readings

  A perfectly regular 15-minute Libre series therefore has 100% coverage. `tests/test_alignment.py` checks this for both sensors.
- **Heart rate and METs** (1-minute) are summarised over windows ending at each reading time.
- **Imputation** fills only missing expected Dexcom readings, in gaps of up to 20 minutes.
  - Imputed values are used for features only, and tagged *imputed*.
  - Labels never use them.
  - Longer gaps stay missing; LightGBM handles missing values natively.
- **Provenance.** Every field has a provenance column: observed, derived, imputed or synthetic.
- **Track isolation.** Each track writes to its own directories. `tests/test_fusion.py` checks that no ID appears in both tracks.

---

## 3. Prediction task

### 3.1 Event

An **excursion** is a run of consecutive observed primary-sensor readings above 180 mg/dL that spans at least 15 minutes — with Dexcom, at least four readings from onset to onset + 15 minutes. The run must have:

- no reading at or below 180 inside it
- no gap longer than 20 minutes

**Timing**

- **Onset** = the first reading of the run. It is known only in hindsight.
- **Confirmation time** = onset + 15 minutes, when persistence is established.
- **Lead times are measured to the onset.**

**Other rules**

- Two runs merge into one event unless glucose stays at or below 180 for at least 30 minutes between them.
- Censored readings are unambiguous against the threshold: 400 counts as above 180, and 40 as below.

180 mg/dL is the upper bound of the standard time-in-range target. Here it defines the task; it is not an emergency or treatment trigger.

### 3.2 Prediction eligibility (live) vs label evaluability (offline)

| Rule | Information allowed | Decides | Conditions |
|---|---|---|---|
| **Prediction eligibility** | Data at or before *t* only | Whether a prediction is made — live, in replay and in the alert engine | • The current Dexcom reading is observed (not imputed) and at or below 180 <br>• No reading above 180 in the last 30 minutes <br>• At least 60 minutes of history, with ≥ 70% of expected readings observed |
| **Label evaluability** | Data after *t* — offline evaluation only | Whether the outcome can be scored | **Positive:** a confirmed onset falls in (*t*, *t* + 120]; data up to *t* + 135 may be used to confirm an onset near the window's end. <br>**Negative:** no onset, ≥ 80% of expected readings in (*t*, *t* + 120] observed, and no run above 180 that a gap leaves unconfirmed. <br>**Unscorable:** anything else. |

- **Unscorable windows** are excluded from scoring, never counted as negative. Their number is reported, together with the confirmed events that fall near them.
- **No mixing.** The alert engine uses prediction eligibility only; label evaluability never touches it.

### 3.3 Target

| Label | When |
|---|---|
| `y = 1` | A confirmed onset falls in (*t*, *t* + 120] |
| `y = 0` | The window is an evaluable negative |
| Unscorable | Neither of the above |

### 3.4 Two views of the same predictions

| View | Which moments | Purpose |
|---|---|---|
| **Rolling** | All eligible moments | Continuous monitoring |
| **Meal-time** | Eligible moments at a logged meal's start | The "will this meal cause an excursion?" view; reported as a subset |

### 3.5 Forecast (secondary)

- The twin's glucose forecast at +30 and +60 minutes is compared against the trend baseline.
- Censored target readings are excluded from forecast error and counted.

---

## 4. Personal twin

### 4.1 Three-parameter model

$$\hat G(t) = b_0 + \sum_{m:\ t_m \le t} k_c\, C_m\, \phi(t - t_m;\ \tau)$$

$$\phi(s;\tau) = \frac{s}{\tau}\, e^{\,1 - s/\tau}, \qquad s \ge 0$$

| Parameter | Meaning | Unit |
|---|---|---|
| $b_0$ | Baseline glucose | mg/dL |
| $k_c$ | Rise at peak per gram of carbohydrate eaten | mg/dL per g |
| $\tau$ | Time to peak | minutes |

$C_m$ is the carbohydrate eaten in meal *m*: logged carbs × proportion eaten. The response curve $\phi$ peaks at height 1 when *s* = τ, then decays.

This is a **phenomenological** model. It does not simulate insulin, digestion or medication. Its parameters are descriptive and are not interpreted physiologically.

The study spaced meals at least 3 hours apart, so individual meal responses overlap little. That makes the fit easier.

### 4.2 Fitting

$$\hat\theta = \arg\min_\theta \sum_{t \in \mathcal{O}} \frac{\big(G(t) - \hat G(t;\theta)\big)^2}{\sigma^2} + \sum_j \frac{\big(\theta_j - \mu_j\big)^2}{s_j^2}$$

- **Data term.** $\mathcal{O}$ is the set of **observed, non-imputed, non-censored** primary readings inside the allowed fitting window.
- **Censored readings.** Readings of 400 are excluded, because treating them as exact would pull the fit down. Each participant's censoring rate is reported. Participants with more than 1% censored readings are flagged; a censored-data loss is postponed (§0).
- **Scale.** $k_c$ and $\tau$ are fitted on a log scale.
- **Solver.** `scipy.optimize.least_squares`.
- **Identifiability check.** Refit on bootstrap resamples of the participant's days and report how much the parameters spread.

### 4.3 Three levels of personalisation

| Level | Prior mean μ | Participant's own data |
|---|---|---|
| **P0 — population default** | Average fitted θ across training participants | None |
| **P1 — profile-informed** | Ridge regression from the lab profile (HbA1c, fasting glucose, HOMA-IR, BMI, age) to θ, learned on training participants | None |
| **P2 — updated with own data** | P1 | The participant's observed glucose and meals up to time *t* |

In replay, the twin is refitted once per day at midnight, using data up to that moment only.

### 4.4 Forecast and uncertainty

**Anchoring.** The forecast starts from the observed glucose. The twin's current error fades with time constant $\tau_r$, fitted in the inner folds:

$$\tilde G(t+h) = \hat G(t+h) + \big(G(t) - \hat G(t)\big)\, e^{-h/\tau_r}$$

- **No future meals.** The forecast assumes no new meals after *t*. This is why the meal-time view matters.
- **Uncertainty band.** An empirical 80% interval: the 10th–90th percentiles of forecast errors at each horizon, from inner-fold participants.

### 4.5 Extensions

| ID | Extension | Rule |
|---|---|---|
| E1 | Protein, fat and fibre damping | $k_c C_m \rightarrow k_c C_m / \big(1 + \beta (P_m + F_m + \mathrm{Fib}_m)\big)$ |
| E2 | Observed post-meal activity | Response scaled by $e^{-k_a E_m(t)}$. $E_m(t)$ = MET-minutes above 1.5 between the meal and min(*t*, meal + 60 min). Only activity already observed counts; nothing is assumed about future activity. |
| E3 | Estimated sleep | Only on nights with adequate coverage (§2.3) |

**Selection rule (inner cross-validation only).**

1. Within the training participants, compute each participant's RMSE difference: RMSE with the extension − RMSE without it.
2. Bootstrap the 95% confidence interval of that difference.
3. Keep the extension only if the whole interval lies **below zero**.

**Protocol.** Selection uses the cold-start protocol of §6.5: the twin is fitted once on each inner-validation participant's first 72 hours and scored from hour 72, with τ_r fixed at its default of 60 minutes.

**Reporting.** Every extension is reported on the outer folds whether it was kept or not, together with how many outer folds selected it.

---

## 5. Event model

### 5.1 Stages

| Stage | Model | Inputs |
|---|---|---|
| **M0** | Trend baseline. Score = current glucose + 60 × 30-minute slope (projected glucose at +60). | Glucose only |
| **M1** | Dynamic LightGBM | Dynamic features |
| **M2** | Fused LightGBM | M1 + lab profile |
| **M3** | Twin-enhanced LightGBM | M2 + twin features |

- **Reporting.** All four are always reported on the outer folds.
- **Selection.** The dashboard model is chosen by inner cross-validation: highest early-warning rate at the same alert budget.
- **Expectation.** Rows from the same participant are highly correlated, so M1–M3 may not beat M0. If so, that is reported as found.

### 5.2 Features (fixed before evaluation)

| Family | Features | Count |
|---|---|---|
| Glucose | Current glucose; slopes over 15, 30 and 60 min; SD over 60 min; max over the last 180 min; minutes since glucose was last above 180 (capped at 360) | 7 |
| Meals | Minutes since last meal (capped at 360); last meal's carbs eaten; carbs eaten in the last 180 min; carbs on board (90-minute decay) | 4 |
| Activity | Mean METs over the last 30 min; observed MET-minutes above 1.5 since the last meal; mean heart rate over the last 30 min | 3 |
| Time | Hour of day as sine and cosine — only if the meal-order check passes | 2 |
| Profile (M2+) | HbA1c, fasting glucose, HOMA-IR, BMI, age | 5 |
| Twin (M3) | Twin forecast at +60 and +120; predicted peak of the current meal response; mean twin residual over the last 30 min | 4 |

Censored readings enter features at their bound, with the censoring rate reported.

### 5.3 Settings (fixed in advance)

LightGBM:

- `num_leaves` 15
- learning rate 0.05
- 300 trees
- `min_child_samples` 100
- feature and bagging fraction 0.8
- balanced class weights
- seed 42

### 5.4 Profile-only analysis (separate)

A lab profile cannot time an excursion. So instead of a profile-only classifier, we report how the profile relates to each participant's overall excursion rate (e.g. rank correlation with HbA1c), labelled **baseline propensity**.

---

## 6. Evaluation

### 6.1 Nested, participant-wise cross-validation

- **Outer loop.** `StratifiedGroupKFold`, 5 folds, stratified by diagnosis group, fixed seed. Used **only for reporting**.
- **Inner loop.** Participant-wise cross-validation within each outer training set. It fits or selects:
  - P0 and P1 priors
  - $\tau_r$
  - forecast-error quantiles
  - alert thresholds
  - twin extensions
  - the dashboard model
- **Fixed in advance:** hyperparameters and imputation rules.
- **Final dashboard model.** The same inner procedure is applied to all participants. Its expected performance is the outer-fold estimate of that whole procedure.
- **One approximation.** Inside the inner loop, twin features for training participants use priors fitted on the whole outer training set, not re-fitted for every inner split. This can only make inner *selection* slightly optimistic; outer-fold estimates are unaffected.

### 6.2 Event-to-alert matching

The rule is deterministic:

1. **Match alerts to events.** Process alerts in time order. Each alert is matched to the **first** confirmed event whose onset falls in (alert time, alert time + 120 min].
   - An alert is matched to at most one event.
   - An event may collect several alerts.
2. **Classify each event:**
   - **Warned early:** its earliest matched alert came ≥ 30 minutes before onset.
   - **Warned late:** every matched alert came less than 30 minutes before onset.
   - **Missed:** no matched alert.

   Lead time = onset − earliest matched alert.
3. **Classify each alert:**
   - **True:** matched.
   - **False:** unmatched, and its 120-minute follow-up window is evaluable.
   - **Unscorable:** unmatched, and its follow-up window is not evaluable.
4. **Tests.** `tests/test_matching.py` covers one alert followed by two events, alerts inside the cooldown, late alerts and unscorable follow-ups.

### 6.3 Metrics

**Primary**, computed per participant and then averaged:

- early-warning rate
- false alerts per patient-day

**Also reported:** late-warning rate, miss rate, median lead time to onset, and unscorable windows and alerts.

**Secondary** (evaluable windows only): row-level AUPRC, AUROC and Brier score.

**Forecast**

- Twin vs trend baseline: RMSE and MAE at +30 and +60 minutes, on non-censored targets
- Clarke error grid (% in zones A+B)

**Always reported**

- Counts: participants in the primary analysis, excluded participants and why, eligible moments, evaluable and unscorable windows, confirmed events
- 95% confidence intervals from a bootstrap over participants (1,000 resamples)
- Per-group results (healthy, prediabetes, Type 2 diabetes), described only

All results are framed as exploratory feasibility evidence, not proof that one model is better.

### 6.4 Alert threshold

1. For each model, choose the threshold in the inner loop, with the 90-minute cooldown applied, so that alerts average at most 3 per patient-day.
2. Freeze it.
3. Apply it unchanged to the outer fold.

### 6.5 Personalisation (cold-start) experiment

For each held-out participant:

| Step | Rule |
|---|---|
| **Fitting window** | The participant's first *k* hours, with *k* ∈ {0, 6, 24, 72}. Only data from this window enters their fit; the P0 and P1 priors come from training participants only. |
| **Evaluation window** | Hour 72 to the end of their record. Identical for every *k*, and it never overlaps the fitting window. |
| **Comparison** | P0 and P1 at *k* = 0, and P2 at each *k* (P2 = P1 when *k* = 0) |
| **Metric** | Twin 60-minute forecast RMSE and MAE, on non-censored readings in the evaluation window |

If the lab profile or the person's own data does not help, the report says so.

### 6.6 Sensitivity analyses

- With vs without imputed feature values
- Libre as the primary sensor, at its native 15-minute cadence
- 5-minute vs 15-minute event persistence — secondary; the primary definition stays fixed
- Excluding windows that contain censored readings

Run them with `--sensitivity`. They compare M0–M2; the twin-enhanced model is evaluated in the primary analysis only.

### 6.7 Reporting

`scripts/run_pipeline.py` writes these files:

- `reports/data_summary.md`
- `reports/results.md`
- `reports/metrics.json`

Scores are never edited by hand.

---

## 7. Alert engine (live rules only)

**An alert fires when** the moment is prediction-eligible and the risk is at or above the frozen threshold.

**No alert fires:**

- within 90 minutes of a previous alert (cooldown)
- when the current reading is above 180 — the status shows "above range" instead
- when fewer than 50% of the expected readings in the last 30 minutes are observed — the status shows "insufficient data" instead

The engine never uses label evaluability or any data after *t*.

**Each alert records:**

- time and risk
- top reasons
- data coverage
- model version

Low-glucose alerts are postponed: there are too few real events to evaluate them.

---

## 8. Explanations

1. LightGBM computes exact TreeSHAP contributions for the dashboard model (`pred_contrib`), so no separate SHAP package is needed.
2. Contributions are summed by feature family.
3. The top families are turned into plain-language sentences using the participant's own values, for example:
   - "Last meal had 72 g of carbohydrate."
   - "Glucose is rising 1.8 mg/dL per minute."

They are labelled **factors the model associated with higher risk**, never causes.

---

## 9. What-if scenarios

**Controls (v1)**

- Change the carbohydrate in the current or next meal (g).
- A hypothetical post-meal walk of *N* minutes at about 3.5 METs. This control appears only if extension E2 was selected.

**Display**

- Baseline and scenario curves side by side
- A list of the inputs that changed
- Uncertainty bands
- Peak and minutes above 180 for both

**Low-evidence warning** appears when either:

- the scenario lies outside this participant's observed range
- the participant has fewer than 5 logged meals

**Fixed text on the screen:** "Simulated estimate from a simple personal meal-response model — not a physiological simulation, a measured effect or a treatment recommendation."

There is no optimisation and no suggested change.

---

## 10. Dashboard

**Tabs**

| Tab | Contents |
|---|---|
| **Panel** | Participants ranked by risk at the replay time; coverage; alerts today |
| **Patient** | Profile (observed; FHIR-derived on the synthetic track); glucose trace with observed and imputed styled differently; forecast and band, labelled as a simple meal-response model; events, meals, activity; alert history |
| **Why** | Plain-language reasons (§8) |
| **What-if** | User-chosen scenarios (§9) |
| **Data quality** | Coverage against each sensor's native interval, imputed share, gaps, censored readings, provenance legend |

- **Badges:** Observed · Estimated · Imputed · Synthetic.
- **Track switch:** evidence or synthetic. A permanent banner marks the synthetic track.
- **Hosted demo:** synthetic track only, because of the CGMacros licence (§12). The evidence track runs locally.
- **Replay rule:** everything on screen is computed from data at or before the replay time. `tests/test_replay.py` deletes all data after time *t* and checks that every output at *t* is unchanged.

---

## 11. Reproducibility and pipelines

- Python 3.11 with pinned dependency versions.
- Fixed seeds everywhere.
- One command runs the whole pipeline. Reports are regenerated from scratch.
- Demo mode runs on the synthetic track, without the raw dataset.
- Before the video is recorded: test in a fresh environment (clone → install → `pytest` → demo).

### 11.1 Research pipeline: what each stage must guarantee

- The two tracks run as separate executions with separate outputs.
- A failed validation check stops the run.
- A source-data assumption is never silently repaired just to let later steps run.

| Stage | Guarantee before moving on |
|---|---|
| 1. Ingestion | Every file is identified and parsed; real and synthetic identifiers never cross |
| 2. Validation and provenance | Native sensor readings are reconstructed; the HbA1c interpretation is justified; joins and quality checks are reported |
| 3. Alignment and labels | Onset and confirmation are distinct; unscorable windows never become negative labels |
| 4. Baseline and twin | M0 is reproducible; the twin fits only permitted readings and uses no future data |
| 5. Training and evaluation | No participant crosses folds; selection happens only in inner folds; all four models are reported |
| 6. Reporting | Counts, event metrics, intervals and forecast errors are generated automatically |
| 7. Dashboard | Replay at time *t* never uses data after *t*; synthetic data is visibly marked |

### 11.2 GitHub CI pipeline

`.github/workflows/ci.yml` checks that the code is reliable and safe to merge. It tests code and data contracts, and it **never needs CGMacros**.

| Step | What runs |
|---|---|
| **Trigger** | Every push and pull request to `main`; a manual trigger for the heavier full synthetic demo run |
| **Environment** | Python 3.11 and pinned dependencies; package install and import check |
| **Static checks** | `ruff check` and `ruff format --check` |
| **Unit tests** | `pytest` on small, deterministic synthetic fixtures shaped like CGMacros (`tests/fixtures/`): validation, alignment, fusion, labels, matching, leakage, twin and replay |
| **Synthetic smoke run** | `python scripts/run_pipeline.py --demo --smoke --out ci-smoke` on a tiny synthetic cohort. Checks that profiles and streams fuse and that every expected output exists. |
| **Repository safety** | `scripts/check_repo_safety.py` fails the run if it finds any of: <br>• tracked files under `data/raw/` or `data/processed/` <br>• CGMacros-named files <br>• participant-level evidence-track tables <br>• model files outside the synthetic demo folder <br>• files over 5 MB <br>• credential-like strings |
| **Artifacts** | Test results and synthetic-only smoke outputs, kept for inspection |
| **Deployment** | Branch protection requires CI to pass before merging to `main`. The hosted demo — synthetic track only — deploys from `main`. |

**Rules**

- The evidence-track evaluation runs locally, after the dataset has been downloaded and validated.
- Its outputs go to git-ignored folders.
- Only aggregate reports may be committed: `reports/data_summary.md`, `reports/results.md` and `reports/metrics.json`.

---

## 12. Privacy, safety, compliance and licences

**Data and safety**

- **No identifiable data.** Synthea identifiers are dropped at ingest. CGMacros is anonymised and not redistributed.
- **Clinical safety.** Decision support for research only: no diagnosis, dosing or emergency use. A disclaimer appears on every screen.

**CGMacros licence: CC BY-NC-SA 4.0**

- **Attribution.** Cite the dataset (PhysioNet, doi:10.13026/3z8q-x658), its *Scientific Data* paper and PhysioNet's standard citation. Link the licence and state the changes we make.
- **NonCommercial.** Use is for non-commercial research and education. Whether a prize competition qualifies is confirmed with the organisers (see §14). Any commercial follow-on would need differently licensed data.
- **ShareAlike.** CGMacros-derived files we distribute — e.g. figures showing participant data — carry CC BY-NC-SA 4.0 with attribution. Processed CGMacros tables and models trained on CGMacros are not committed.
- **Our code.** Apache-2.0 covers our code. It does not change the data's licence.

**Path to deployment** (out of scope here)

- clinical validation on Indian patients
- regulatory review as software as a medical device (CDSCO)
- DPDP Act compliance: consent, purpose limitation, minimisation, retention limits, audit logs
- integration with ABDM's FHIR-based health records

---

## 13. Milestones

Each milestone ends with an exit condition. The next one starts only when that condition is met.

**Status (20 Oct 2026):** milestones 0 and 2–6 are implemented and their tests pass on synthetic data. Milestone 1 is implemented and tested on CGMacros-shaped fixtures; its checks on the real files run the first time the pipeline meets CGMacros. Milestone 7 (packaging, PDFs, video) remains.

| # | Milestone | Main files | Exit condition |
|---|---|---|---|
| 0 | **Repository foundation** | `pyproject.toml`, `requirements.txt`, `config.yaml`, `.gitignore`, `.github/workflows/ci.yml`, `scripts/check_repo_safety.py`, `tests/fixtures/` | • A fresh clone installs and passes CI. <br>• Small synthetic fixtures shaped like CGMacros let every test run without the real data. <br>• The safety check blocks raw data, participant-level tables, large files and secrets. |
| 1 | **Verify and ingest the real dataset** | `scripts/download_cgmacros.sh`, `validate.py`, `ingest_cgmacros.py`, `align.py`, `data/README.md`, `docs/data_dictionary.md`, `tests/test_validation.py`, `tests/test_alignment.py`, `tests/test_fusion.py` | • Native readings are reconstructed for both sensors. <br>• Interval-aware coverage passes its tests. <br>• The HbA1c conflict is documented and the group check passes. <br>• Censoring is counted. <br>• The licence and citations are recorded. <br>• Each participant's labs and sensor data join by ID. <br>• `reports/data_summary.md` is generated. |
| 2 | **Labels and a leakage-safe baseline** | `labels.py`, `features.py`, `models.py` (M0), `evaluate.py` (core + matching), `report.py`, `tests/test_labels.py`, `tests/test_matching.py`, `tests/test_no_leakage.py` | • Eligibility and evaluability are separate and tested. <br>• Unscorable counts are reported. <br>• Alert matching passes its tests. <br>• M0 evaluation is reproducible and traceable to data at or before *t*. |
| 3 | **Personal twin** | `twin.py`, `tests/test_twin.py` | • Fits use observed, non-censored readings only. <br>• Fitted curves are sensible and gaps are handled. <br>• The P0 → P1 → P2 comparison runs, and its result is reported either way. |
| 4 | **Compare models** | `models.py` (M1–M3), `alerts.py`, `evaluate.py` (nested CV), `scripts/run_pipeline.py` | • All models and extensions are reported on outer folds; selection happens only in inner folds. <br>• `reports/results.md` is generated end-to-end with no manual editing. |
| 5 | **Synthetic track** | `scripts/generate_synthea.sh`, `ingest_fhir.py`, `simulate.py`, `data/synthetic/` | • Synthetic FHIR records and streams fuse by synthetic ID. <br>• Demo mode runs without the raw dataset. |
| 6 | **Dashboard** | `app/dashboard.py`, `explain.py`, `whatif.py`, `tests/test_replay.py` | • Every displayed value matches the selected participant and replay time. <br>• The truncation test passes. <br>• The hosted demo uses the synthetic track only. |
| 7 | **Package and demonstrate** | README status and results updated from `reports/`, `docs/model_card.md`, `docs/architecture.pdf`, `docs/presentation.pdf`, screenshots, video | • A fresh clone runs the tests and demo mode using only the documented commands. <br>• README claims match the repository's contents. |

---

## 14. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Native sampling misread (1-minute rows vs 5/15-minute sensors) | Native-reading reconstruction is the first validation step; interval-aware coverage tests |
| Too few participants with adequate Dexcom coverage | Exclusions counted; Libre sensitivity analysis at 15-minute cadence |
| Few participants (45) and possibly few events or many unscorable windows | All counts reported; participant bootstrap intervals; exploratory framing |
| Optimistic results from model or extension selection | Nested cross-validation; every model and extension reported |
| Censored readings bias the twin | Excluded from the twin's loss and from forecast error; rates reported |
| HbA1c units wrong | Interpreted as percent only if the published groups reproduce; otherwise stop |
| CGMacros licence is non-commercial | Confirm with the organisers that a prize-competition entry is acceptable; hosted demo uses the synthetic track; no CGMacros-derived models or tables committed |
| Synthetic data mistaken for real | Isolated runs, provenance tags, badges, permanent banner |
| What-if read as advice | User-chosen scenarios only; low-evidence warnings; fixed disclaimer |
| US participants and US-style meals | Stated limitation; Indian scenarios confined to the synthetic track |

---

## Revision notes

**v2 → v3**

| Area | v2 | v3 |
|---|---|---|
| Sensor sampling | One 5-minute grid; primary sensor chosen by coverage | Native readings reconstructed from 1-minute rows; Dexcom (5-minute) primary; coverage interval-aware; Libre kept for cross-checks and sensitivity |
| Eligibility | One rule mixing past and future data | Live prediction eligibility (past only) vs offline label evaluability; unscorable windows reported |
| Alerts vs events | Partly defined | Deterministic matching; early, late and missed events; true, false and unscorable alerts |
| Model selection | Kept if it beat the simpler model | Nested cross-validation (select inside, report outside); every model and extension reported |
| Censored readings | Flagged | Excluded from the twin's loss and from forecast error; counted |
| HbA1c | Treated as percent if in range | Percent only if published groups reproduce; conflict documented |
| Licence | "Terms on the PhysioNet page" | CC BY-NC-SA 4.0 recorded, with handling rules |
| Architecture | Both tracks into one block | Two isolated runs |
| README | Claimed requirements "met" | Planned coverage with status; deliverables listed as planned until they exist |

**v3.1 → v3.2 (implementation):** built the pipeline as specified. Changes made while building:

- explanations use LightGBM's built-in TreeSHAP;
- extension selection uses the cold-start protocol with τ_r fixed at 60 minutes;
- inner-loop twin features use the outer-training priors (stated in §6.1);
- sensitivity analyses compare M0–M2 and run with `--sensitivity`;
- the personalisation table reports P0 and P1 starting points across hours of own data;
- Synthea labs that are implausible are replaced by simulated values and tagged;
- the demo evaluation uses a 60-patient subset.

**v3 → v3.1:** added the research-pipeline stage guarantees and the GitHub CI pipeline (§11), and milestone 0 (repository foundation).

**v1 → v2:** separated real and synthetic data; cut the twin to 3 parameters; made metrics event-level; staged the models; scoped the what-if.
