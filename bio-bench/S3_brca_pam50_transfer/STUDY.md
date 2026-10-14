# S3 · BRCA PAM50 subtype transfer, TCGA → CPTAC

**Role:** the cross-cohort study.**Both cohorts have their own published PAM50 labels**, so training and external evaluation are independent. It is a multiclass study on built-in choices, plus custom instructions for the harmonisation variants.

**Needs E1** (external cohort) in MLCA v2.


## Data and preparation (`prepare_data.py`, to be written; non-AI)

| Input | Content |
|---|---|
| `${SOURCE_DATA}/brca_tcga_pan_can_atlas_2018/data_mrna_seq_v2_rsem.txt` | TCGA-BRCA RNA (genes × samples; `Hugo_Symbol`, `Entrez_Gene_Id`) |
| `${SOURCE_DATA}/brca_tcga_pan_can_atlas_2018/data_clinical_patient.txt` | `SUBTYPE`: BRCA_LumA 499, BRCA_LumB 197, BRCA_Basal 171, BRCA_Her2 78, BRCA_Normal 36, missing 103 |
| `${SOURCE_DATA}/brca_cptac/HS_CPTAC_BRCA_2018_RNA_GENE.cct`, `HS_CPTAC_BRCA_2018_CLI.tsi` | CPTAC-BRCA RNA (122 samples) and `PAM50` |

The script must:
1. Map TCGA primary-tumour samples (`-01`) to patients.
2. Map the labels `BRCA_LumA` → `LumA`, and so on.
3. Intersect the gene symbols, keeping the first occurrence of a duplicated symbol (recorded).
4. Transform TCGA RSEM with `log2(x+1)`, a fixed transform, not data-dependent; CPTAC as provided.
5. Write `tcga_pam50.csv` (discovery) and `cptac_pam50.csv` (external) to `data_prep/S3/`, with SHA-256 and counts.

**Normal class:** absent in CPTAC. **Decision:** drop Normal from training (recommended), or keep it and exclude it from the external metrics.

## Protocol (template `tabular_classification`, multiclass; TCGA = discovery, CPTAC = external via E1)

| # | Event | Built-in | Custom instruction | Options and notes |
|---|---|---|---|---|
| 1 | onboard | register `tcga_pam50.csv` (primary) and `cptac_pam50.csv` (role `external`); target `PAM50`; unit "one tumour" | — | — |
| 2 | data | `audit_schema`; `count_target`; data review (multiclass → "predict all subtypes") | — | — |
| 3 | eda | `describe_distributions` | CI-1: "PCA of the training-eligible TCGA rows coloured by PAM50." (no external data before unlock) | A joint PCA with CPTAC is possible only after the unlock (step 9) |
| 4 | split | `freeze_stratified` on TCGA 60/20/20 | — | The CPTAC external cohort stays sealed |
| 5 | preprocess | `fit_preprocessing` (scale fitted on TCGA train) | CI-2 (variants, D-S3): (a) none: train-only scaling, the primary and clean option; (b) "per-cohort z-scaling" (label-free, uses external feature statistics; allowed only after unlock, at step 9); (c) "ComBat with batch {TCGA, CPTAC}" (label-free, uses external features; step 9) | Each variant is labelled; (b) and (c) are transductive and reported as such |
| 6 | features | `select_features` (k chosen on train) or `retain_features` | — | Recommended: retain |
| 7 | model | `train_logistic` (multinomial, class-balanced); `train_alternative` | CI-3: "Also fit a calibrated linear SVM and a random forest (class-balanced)." | Selection on the TCGA validation split |
| 8 | validate | `score_validation` (multiclass ROC AUC one-vs-rest, balanced accuracy); approval | CI-4: "Macro-F1, macro-AUPRC and per-class PR curves on validation rows." | — |
| 9 | test (+ external) | unlock; `score_test` on TCGA test | CI-5 (needs E1): "Score the frozen model once on the external CPTAC cohort: accuracy, macro-F1, balanced accuracy, macro-AUPRC, per-class AUPRC and confusion matrix; repeat for harmonisation variants (b) and (c)." | One exposure per variant, all reported |
| 10 | interpret | `explain_model` | — | — |
| 11 | report | `generate_reports` | — | — |


## Runs

**Not possible until E1 exists.** Then: `worker` ≥ 5 models.
