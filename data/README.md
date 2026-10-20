# Data

GlucoTwin uses two tracks that never mix: real, anonymised **CGMacros** data (evidence track,
all reported metrics) and **synthetic** data (Synthea EHRs plus simulated wearables; demo only).

| Folder | Contents | In Git? |
|---|---|---|
| `raw/cgmacros/` | CGMacros, downloaded by `scripts/download_cgmacros.sh` | **No** (git-ignored) |
| `processed/<track>/` | Aligned tables written by the pipeline | **No** (git-ignored) |
| `synthetic/synthetic_profiles.csv` | 200 synthetic patient profiles built from Synthea FHIR records | Yes |
| `synthetic/synthea_run.json` | How the Synthea run was made (version, seed, counts) | Yes |
| `synthetic/fhir_sample/` | Three trimmed Synthea FHIR bundles, without names or addresses | Yes |
| `synthetic/synthea_raw/` | Raw Synthea output (about 2.4 GB) | **No** (git-ignored) |

## CGMacros (evidence track)

| | |
|---|---|
| Dataset | CGMacros: a scientific dataset for personalized nutrition and diet monitoring, version 1.0.0 |
| Host | PhysioNet, https://physionet.org/content/cgmacros/1.0.0/ (open access) |
| Licence | **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International** (CC BY-NC-SA 4.0), https://creativecommons.org/licenses/by-nc-sa/4.0/ |
| Contents | 45 adults (15 healthy, 16 prediabetes, 14 Type 2 diabetes), 10 days each: Dexcom G6 Pro CGM (5-minute sampling), FreeStyle Libre Pro CGM (15-minute sampling), Fitbit heart rate and METs, logged meals with macronutrients, lab profile |

### Access

```bash
bash scripts/download_cgmacros.sh          # needs wget or curl, about 630 MB
```

Or download `CGMacros_dateshifted365.zip` from the PhysioNet page and unzip it into
`data/raw/cgmacros/`. The loader searches that folder recursively for `bio.csv` and the
`CGMacros-0XX.csv` files.

### How this project meets the licence

- **Attribution.** We cite the dataset, its paper and PhysioNet (below), link the licence, and
  state our changes. Changes: native sensor readings are reconstructed from the 1-minute rows;
  readings are aligned per participant; LDL = 800 and VLDL = 400 error codes are set to missing;
  METs are divided by 10; HbA1c is read as percent after the group check; derived fields
  (HOMA-IR, triglyceride/HDL, diagnosis group, heuristic sleep) are added.
- **NonCommercial.** Used for non-commercial research and education. Any commercial follow-on
  would need differently licensed data. Using the dataset in a prize competition should be
  confirmed with the organisers.
- **ShareAlike.** CGMacros-derived files we distribute (for example a figure showing participant
  data) carry CC BY-NC-SA 4.0 and this attribution. Processed CGMacros tables and models trained
  on CGMacros are never committed (`scripts/check_repo_safety.py` enforces this). The hosted demo
  uses the synthetic track only.
- Our Apache-2.0 licence covers our code only; it does not change the data's licence.

### Citations

1. Gutierrez-Osuna R, Kerr D, Mortazavi B, Das A. CGMacros: a scientific dataset for personalized
   nutrition and diet monitoring (version 1.0.0). PhysioNet, 2025. RRID:SCR_007345.
   https://doi.org/10.13026/3z8q-x658
2. Das A et al. CGMacros: a scientific dataset for personalized nutrition and diet monitoring.
   *Scientific Data*, 2025. doi:10.1038/s41597-025-05851-7
3. Pollard T, Moody BE, Lehman L, Gow B, Fernandes C, Xie C, Johnson A, Mark RG, Heldt T.
   PhysioNet as a global platform for biomedical research. *Nature Health*, 2026.
   https://doi.org/10.1038/s44360-026-00096-z

### Known quirks the pipeline checks (stop rules in APPROACH.md §2.1)

- Rows are 1-minute, while Dexcom samples every 5 minutes and Libre every 15. The pipeline
  detects whether the rows in between are blank, carried forward or interpolated, and stops if
  it cannot tell.
- The data dictionary labels HbA1c as mmol/mol, but the values (4.6–8.5) and the dataset's group
  thresholds are in percent. Percent is used only if the published group sizes (15/16/14)
  reproduce.
- Dates are shifted per participant by whole days, so dates are never compared across people.

## Synthetic track

Generated with **Synthea v3.4.0** (`scripts/generate_synthea.sh`): 1,000 adults aged 25–75,
seed 42, reference date 2026-10-01, FHIR R4 export, 5 years of history. `synthea_run.json`
records the counts.

- **Profiles.** Living adults aged 25–75 are selected to a mix of 80 Type 2 diabetes,
  80 prediabetes and 40 normoglycaemic patients. Selection is weighted toward an approximate
  Indian adult age mix that we set in `config.yaml`. Groups come from Synthea's conditions.
- **Labs.** HbA1c and fasting glucose come from Synthea when present and plausible; otherwise they
  are simulated from group-specific ranges.
- **Simulated by us.** Fasting insulin (Synthea does not model it) and a TCF7L2 risk-allele count
  are always simulated. Every field's origin is in its `prov_*` column.
- **Wearable streams** are simulated at run time by `src/glucotwin/simulate.py` (deterministic for
  the configured seed): CGM, heart rate, activity, sleep and HRV.
- **Indian meal library.** Meals use the library in `config.yaml`. Its per-serving macros are
  approximate standard-recipe estimates chosen by us; check them against the Indian Food
  Composition Tables 2017 (Longvah T, Ananthan R, Bhaskarachary K, Venkaiah K; National Institute
  of Nutrition, ICMR, Hyderabad) before relying on them.

Synthetic data demonstrates the pipeline and the simulated-EHR fusion the challenge asks for. Its
glucose comes from a meal-response model like the twin, so it is **never** used as evidence of
accuracy.
