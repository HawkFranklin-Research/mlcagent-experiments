# S1 · CBC anemia classification

**Role in the revised paper:** the first and simplest study. It runs almost entirely on MLCA's built-in choices (guided autopilot), with 2–3 custom instructions for figures the template does not produce.


## Data

| Item | Value |
|---|---|
| File | `${SOURCE_DATA}/final results/hfr/Anemia/anemia.csv` (compare SHA-256 with the author's other copy before the runs) |
| Shape | 1,421 rows × 6 columns: `Gender`, `Hemoglobin`, `MCH`, `MCHC`, `MCV`, `Result` (0 = 801, 1 = 620) |
| Preparation | none (registered as is) |
| Source | to be stated by the author (dataset citation) |

## Protocol (template `tabular_classification`; built-in = option card, CI = custom instruction)

| # | Event | Built-in | Custom instruction | Options and notes |
|---|---|---|---|---|
| 1 | onboard | study details: target `Result`, unit "one individual", repeated subjects "no"; AI-drafted brief, author approves | — | — |
| 2 | data | `audit_schema`; `count_target`; **data review** | — | Expected question: *Hemoglobin separates the classes (almost) perfectly — exclude?* (explicit answer required). **D-S1:** primary = keep and flag; sensitivity = a **branch** from `data` that excludes Hemoglobin, to show how much the label depends on it |
| 3 | eda | `describe_distributions`; `plot_missingness` | CI-1: "Pairplot of Hemoglobin, MCV, MCH and MCHC coloured by Result; save one figure." | — |
| 4 | split | `freeze_stratified` 60/20/20, seed 42 | — | — |
| 5 | preprocess | `fit_preprocessing` (median impute + standard scale; one-hot `Gender`; fitted on train) | — | — |
| 6 | features | `retain_features` | — | 5 features; no selection |
| 7 | model | `train_logistic` (baseline); `train_alternative` (HistGradientBoosting) | CI-2 (optional): "Also fit random forest, RBF-SVM and extra trees on training rows with the same preprocessing." | — |
| 8 | validate | `score_validation` (ROC AUC, PR AUC, balanced accuracy, Brier, Youden threshold); `audit_validation`; **approval:** model + threshold | CI-3: "Calibration curves (10 bins) for all candidates and a learning curve (F1 vs training size, 5 sizes) for logistic regression, on training/validation rows." | — |
| 9 | test | **unlock** (one exposure); `score_test` (bootstrap 95 % CI) | — | — |
| 10 | interpret | `explain_model` (permutation importance); `plot_results` | CI-4: "SHAP summary (beeswarm) for the frozen model on validation rows." (needs package approval `shap`) | — |
| 11 | report | `generate_reports` | — | — |

**Expected custom instructions:** 3–4. Everything else runs on recommended choices.


## Outputs per run

These go into `results/<model-slug>/run-NNN/`, as laid out in `../README.md`: scripts for every step, custom instructions, reports, figures (pairplot, calibration, learning curve, SHAP, ROC), `metrics.json`, `usage.csv` and `bench.json`.

## Runs

**Today:** a `mlca bench` card with `guided: true` and the approvals as policy. CI-1–CI-4 need E2 to run unattended; until then, issue them in the TUI and record them.

**Arms:**
- `worker`: AI-written code, ≥ 5 models;
- `reference`: the no-AI baseline, 1 run.
