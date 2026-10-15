# S4 · HNSCC subtype discovery → prognostic signature → CPTAC validation

> **15 October 2026:** this study is part of the evaluation (Phase 2) in `../../README.md`. The frozen agent receives the prepared data and a short starting brief; the scripted step-by-step protocol below is stopped, while its data preparation and aims remain.

**Role:** the last and most custom study. MLCA runs the guided workflow up to a Cox baseline. The analysis then continues as a **sequence of custom instructions** (clustering, derived labels, repeated classifier screening, survival validation), each one checked against its event's scope, logged, and run as an AI-written script. It shows what custom instructions add to the guided workflow, and it is the flagship `bio-bench` card.

**Needs E1** (external cohort) in MLCA v2.


## Data and preparation (`prepare_data.py`, to be written; non-AI)

| Input (`${SOURCE_DATA}/hnscc_mlca/data/Case-Study-HNSCC-ML-Copilot-Agent/processed_data/`) | Content |
|---|---|
| `tcga.survival.fcptac.csv` | 415 TCGA patients × 517 genes (`PatientID` + genes; values pre-processed by the author in March 2025) |
| `survival_tcga.csv` | `PatientID`, `Overall Survival Status`, `Overall Survival (Months)` |
| `cptac.survival.fcptac.csv`, `survival_cptac.csv` | 102 CPTAC patients, same columns |

The script joins expression and survival per cohort, writes `tcga_hnscc.csv` (discovery) and `cptac_hnscc.csv` (external) with SHA-256 and counts (415/102), and checks that the gene columns are identical.

**Stated limitation:** the input matrices were already transformed on whole cohorts before any split, by the data provider's preprocessing. A cleaner variant would rebuild them from `hnscc_data/` (decision, optional).

## Protocol (template `survival`; TCGA = discovery, CPTAC = external via E1)

**Part A: built-in choices**

| # | Event | Built-in | Notes |
|---|---|---|---|
| A1 | onboard | duration `Overall Survival (Months)`, event `Overall Survival Status`; unit "one patient"; repeated "no" | AI-drafted brief, author approves |
| A2 | data | `audit_schema`; data review (`PatientID` → exclude) | — |
| A3 | eda | `describe_distributions` | — |
| A4 | split | `freeze_random` (or stratified by event) 60/20/20, seed 42 | Patient-level; test sealed |
| A5 | preprocess | `fit_preprocessing` (train-fitted) | — |
| A6 | features | `retain_features` | — |
| A7 | model | `train_cox` (penalised Cox baseline) | Gives a C-index baseline |
| A8 | validate | `score_validation` (C-index) | — |

**Part B: custom-instruction deep dive** (in the `model` and `validate` events, which may read training and validation rows and prior outputs; the `features` event's scope forbids fitting models)

| # | Event | Custom instruction (draft wording) | Decision |
|---|---|---|---|
| B1 | model | "Cluster the **training** patients' expression with KMeans, agglomerative (Ward) and Gaussian mixture for k = 2…10 (seed 42). For each, record silhouette, Calinski–Harabasz, Davies–Bouldin, KMeans inertia and the multi-group log-rank p on training survival. Save the table and the Kaplan–Meier plots." | **D17-2:** training patients (b), or all TCGA patients (a, possible only before the split, in `eda`, and then labelled transductive) |
| B2 | model | "Select one configuration by <rule>, refit it, and assign every training patient to a cluster; assign validation and test patients with the fitted model (predict, or nearest centroid for agglomerative). Label the two largest clusters 0/1; others are excluded from classification. Save the assignments." | **D17-1:** lowest training log-rank p, or best silhouette (prespecified), or both, labelled |
| B3 | model | "On training patients with labels 0/1: for each feature size in {2, 5, 8, 10, 12, 15, 20} and each classifier (LR, RF, SVM, KNN, GB): repeat N stratified 70/30 splits of the training patients; fit SelectKBest (f_classif) and the classifier on each inner training part only; record ROC AUC on each inner held-out part and the selected genes. Save all runs and a summary (mean ± SD AUC per classifier × size)." | **D17-3:** N = 100 or 50 |
| B4 | model | "Choose the signature by <rule> (highest mean AUC; ties → smaller size), refit selector and classifier on all training patients, and save the gene list and the model." | — |
| B5 | validate | "Apply the signature model to validation patients: predicted groups, Kaplan–Meier, log-rank p, hazard ratio (Cox on the predicted group), and C-index of the predicted probability." | approval: freeze signature |
| B6 | test (after unlock) | "Apply the frozen signature once to the TCGA test patients, then to the external CPTAC cohort (genes intersected): Kaplan–Meier, log-rank p, hazard ratio with 95 % CI, group sizes." | needs E1 |
| B7 | interpret | "Rank the signature genes by importance; Kaplan–Meier panel of training, validation, test and CPTAC side by side." | — |
| B8 | report | `generate_reports` (built-in) | — |

**Expected custom instructions:** 7, at least 2–3 sentences each, recorded with their word counts. The Part A choices are recommended defaults.


## Runs

Not possible end to end until E1 (and E2 for unattended runs). Part A and B1–B5 can be run today in the TUI as custom instructions. Arms: `worker` ≥ 5 models.
