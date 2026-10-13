# S2 · CPTAC-BRCA receptor status: RNA vs proteome

**Role:** a second guided study, on high-dimensional omics. It has 6 tasks (ER, PR, HER2 × RNA, proteome) run as 6 projects. A few custom instructions add cross-validation on training rows and decision curves.


## Data and preparation (`prepare_data.py`, to be written; non-AI, documented)

| Input (under `${SOURCE_DATA}/brca_cptac/`) | Content |
|---|---|
| `HS_CPTAC_BRCA_2018_RNA_GENE.cct` | 23,121 genes × 122 samples |
| `HS_CPTAC_BRCA_2018_Proteome_Ratio_Norm_gene_Median.cct` | 12,621 proteins × samples (with missing values) |
| `HS_CPTAC_BRCA_2018_CLI.tsi` | 122 samples; columns `ER.Updated.Clinical.Status`/`ER`, `PR`, `ERBB2.Proteogenomic.Status`/`Her2.Updated.Clinical.Status`/`HER2.Amplified` (row 2 holds types) |

The script must:
1. Drop the type row.
2. Choose each label column in this fixed order: ER = `ER.Updated.Clinical.Status`, else `ER`; PR = `PR`; HER2 = `ERBB2.Proteogenomic.Status`, else `Her2.Updated.Clinical.Status`, else `HER2.Amplified`.
3. Transpose each matrix to samples × features.
4. Keep the samples with that label.
5. Write 6 CSVs (`<modality>_<receptor>.csv`: `sample_id`, features…, `label`) to ``data_prep/`S2/`.
6. Record the input and output SHA-256 and the counts. Expected counts: ER 81/39, PR 68/47, HER2 15/107.

No scaling, filtering or imputation happens here: those are data-dependent and belong after the split, inside MLCA.

## Protocol per task (template `tabular_classification`)

| # | Event | Built-in | Custom instruction | Options and notes |
|---|---|---|---|---|
| 1 | onboard | target `label`; unit "one tumour sample"; repeated subjects "no" | — | — |
| 2 | data | `audit_schema`; `count_target`; data review | — | Expected questions: `sample_id` is an identifier → exclude; proteome columns > 50 % missing → exclude. Answers recorded. |
| 3 | eda | `describe_distributions` (summary only) | — | Wide table; the summary is capped by the output contract |
| 4 | split | `freeze_stratified` 60/20/20, seed 42 | — | HER2 has 15 positives, about 3 in each of validation and test. **D-S2:** see step 8 |
| 5 | preprocess | `fit_preprocessing` (median impute + scale, train-fitted) | CI-1: "Before imputation, drop features with more than 20 % missing values, with the fraction computed on training rows only." | — |
| 6 | features | `select_features` (SelectKBest, k = min(20, n)) or `retain_features` | — | **Recommended:** retain |
| 7 | model | `train_logistic` (L2, class-balanced) | CI-2 (extensions): "Also fit a linear SVM, a random forest and an elastic-net logistic regression on training rows." | — |
| 8 | validate | `score_validation` (ROC AUC, **PR AUC primary**, Brier, threshold); approval | CI-3: "On training+validation rows only, repeated stratified 5-fold CV (5 repeats) with the full pipeline inside each fold; report mean PR AUC, ROC AUC and Brier per model." (test stays sealed). CI-4: "Calibration curve and decision-curve analysis (net benefit vs treat-all/none, thresholds 0.05–0.95) on validation rows." | **D-S2:** report the single split, the CV, or both (recommended: both, labelled) |
| 9 | test | unlock; `score_test` | — | small n: CIs reported |
| 10 | interpret | `explain_model`; `plot_results` | — | — |
| 11 | report | `generate_reports` | — | — |

**Cross-task summary:** a script (no AI) builds the RNA vs proteome table across the 6 tasks from the 6 `metrics.json` files. 


## Runs

**Today:** 6 `mlca bench` cards (one per task); CI-1–CI-4 via TUI until E2 exists.

**Arms:**
- `worker`: ≥ 5 models;
- `reference`: 1 run.
