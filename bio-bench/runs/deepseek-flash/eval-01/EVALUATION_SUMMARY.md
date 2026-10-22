# Biomedical evaluation, arm A — run summary

Protocol: frozen MLCA `freeze-v1`, direct DeepSeek `deepseek-flash`, reasoning effort `low`; 9 approved tasks × 3 replicates. All failed outcomes are retained. This is a descriptive run record, not a claim of scientific validity for incomplete workflows.

## Results by replicate

| Task / replicate | Completed | Furthest step | Failure category | Setup agrees | Leakage clean | Repairs | Test metric | Test predictions saved | Wall time (s) |
|---|---:|---|---|---:|---:|---:|---|---:|---:|
| `s1_anemia-001` | yes | 12 report_generate_reports_MM39PEY8 | — | True | True | 1 | roc_auc=0.541; pr_auc=0.5059; balanced_accuracy=0.5079; brier_score=0.2937; sensitivity=0.24; specificity=0.7759 | yes | 499.8 |
| `s1_anemia-002` | no | 09 validate_score_validation_ZGS6S2F7 | gate_blocked | True | True | 0 | — | no | 358.5 |
| `s1_anemia-003` | no | 09 validate_score_validation_XEBNP8F1 | gate_blocked | True | True | 1 | — | no | 331.1 |
| `s2_rna_er-001` | no | — | provider_error (HTTP 400) | — | — | 0 | — | no | 45.1 |
| `s2_rna_er-002` | no | — | provider_error (HTTP 400) | — | — | 0 | — | no | 47.3 |
| `s2_rna_er-003` | no | — | provider_error (HTTP 400) | — | — | 0 | — | no | 45.9 |
| `s2_rna_pr-001` | no | — | provider_error (HTTP 400) | — | — | 0 | — | no | 43.9 |
| `s2_rna_pr-002` | no | — | provider_error (HTTP 400) | — | — | 0 | — | no | 45.4 |
| `s2_rna_pr-003` | no | — | provider_error (HTTP 400) | — | — | 0 | — | no | 49.8 |
| `s2_rna_her2-001` | no | — | provider_error (HTTP 400) | — | — | 0 | — | no | 53.0 |
| `s2_rna_her2-002` | no | — | provider_error (HTTP 400) | — | — | 0 | — | no | 49.4 |
| `s2_rna_her2-003` | no | — | provider_error (HTTP 400) | — | — | 0 | — | no | 47.9 |
| `s2_proteome_er-001` | no | — | provider_error (HTTP 400) | — | — | 0 | — | no | 25.8 |
| `s2_proteome_er-002` | no | — | provider_error (HTTP 400) | — | — | 0 | — | no | 31.7 |
| `s2_proteome_er-003` | no | — | provider_error (HTTP 400) | — | — | 0 | — | no | 27.6 |
| `s2_proteome_pr-001` | no | — | provider_error (HTTP 400) | — | — | 0 | — | no | 28.3 |
| `s2_proteome_pr-002` | no | — | provider_error (HTTP 400) | — | — | 0 | — | no | 28.6 |
| `s2_proteome_pr-003` | no | — | provider_error (HTTP 400) | — | — | 0 | — | no | 29.2 |
| `s2_proteome_her2-001` | no | — | provider_error (HTTP 400) | — | — | 0 | — | no | 29.6 |
| `s2_proteome_her2-002` | no | — | provider_error (HTTP 400) | — | — | 0 | — | no | 30.3 |
| `s2_proteome_her2-003` | no | — | provider_error (HTTP 400) | — | — | 0 | — | no | 30.9 |
| `s3_pam50-001` | no | — | provider_error (HTTP 400) | — | — | 0 | — | no | 52.5 |
| `s3_pam50-002` | no | — | provider_error (HTTP 400) | — | — | 0 | — | no | 47.8 |
| `s3_pam50-003` | no | — | provider_error (HTTP 400) | — | — | 0 | — | no | 55.0 |
| `s4_hnscc-001` | no | 09 validate_score_validation_R2X2E0K4 | runtime_error:AttributeError | False | True | 5 | — | no | 758.8 |
| `s4_hnscc-002` | no | 09 validate_score_validation_H25V5FH2 | runtime_error:AttributeError | False | True | 2 | — | no | 624.8 |
| `s4_hnscc-003` | no | 04 split_freeze_random_RFYY7E43 | gate_blocked | False | — | 0 | — | no | 118.2 |

## Totals

- Workflows completed: 1/27 (3.7%).
- HTTP 400 provider failures during setup: 21/27; the recorded artifacts do not include a provider response body, so the underlying 400 cause is undetermined.
- Requests: 182 total; model counts: `{"deepseek-flash": 161, "unavailable": 21}`.
- Known token totals (partial): input 604416, output 479999, reasoning 407007. Token counts are unknown for 21 input, 21 output, and None reasoning records.
- Cost: unavailable for all requests; no dollar total can be substantiated from this batch.
- Test predictions were saved only for `s1_anemia-001`; that run completed all 12 steps. Its test metrics: ROC-AUC 0.541, PR-AUC 0.5059, balanced accuracy 0.5079, Brier score 0.2937, sensitivity 0.24, specificity 0.7759.
- Other runs have no test metric because they did not reach test scoring.
- Reopening rule: applies. 26/27 runs (96.3%) were incomplete, setup-disagreeing, or otherwise unsuccessful, exceeding the 50% threshold. Any new development must follow the preregistered reopening procedure; this batch itself is not rerun or repaired.

## Failure causes (diagnosed 2026-10-22 from the run records)

The batch is not rerun or repaired; these causes inform the reopening (STUDY_DESIGN.md rule 7).

| Runs | Cause | Kind | Evidence |
|---|---|---|---|
| S2 (18), S3 (3): HTTP 400 at setup | The setup request carries a per-column summary of every column. The tables have 12,623–23,123 columns, so the single setup request was 3.5–6.2 MB (S2 RNA 6.15 MB, S2 proteome 3.51 MB, S3 5.25 MB), far beyond the model's input limit; the provider rejected it before any analysis. S4 (520 columns, 0.17 MB) and S1 (6 columns) passed this step. The development tables had at most 65 columns, so no rail or check met this case. | engine (MLCA limitation) | `studies/<run>/attempt-001/setup-001/.mlca/egress.jsonl` (`bytes_sent`); `src/mlca/autonomous/inference.py` `inventory()` at freeze-v1 |
| S1-001: completed, test ROC-AUC 0.541 | The setup proposal listed `Hemoglobin` as a leakage suspect, so it was removed from the features (fitted columns: Gender, MCH, MCHC, MCV). Without Hemoglobin the label is not predictable. A defensible agent decision; under the preregistration (Hemoglobin stays a feature, no leakage columns declared) the run still counts as setup-agreeing and leakage-clean. | agent decision | setup proposal in `setup-001/.mlca/llm/`; `run-001/tables/11_*/interpretation.csv` |
| S1-002, S1-003: gate blocked at validation | Both kept Hemoglobin and reached validation ROC-AUC 0.999 (logistic) and 1.000 (alternative). The Worker recorded the primary metric only under per-model names (`roc_auc:logistic`, `roc_auc_07_logistic`, …), not as `roc_auc`; the engine's check found no primary metric after its scoring-only retry and stopped. | Worker (generic, rail candidate) | `run-001/tables/09_*/validation_metrics.csv`; `run-001/.mlca/steps/09_*/attempts/*/script.py` |
| S4-001, S4-002: AttributeError at validation | The validation script loaded the saved survival model bundle (a dict) and called `predict_partial_hazard` on it. | Worker, survival (outside the rails) | `run-001/.mlca/log` stderr, script line 130 |
| S4-003: gate blocked at the data review | "Choose the outcome column before reviewing the table" after the split, although the setup named duration and event correctly. | engine, survival (to investigate) | `run-001/.mlca/log`; `src/mlca/engine/data_review.py:75` |

The HTTP 400 response bodies were not recorded (`usage.jsonl` keeps the status code only). The cause above follows from the request sizes, not from a provider message.

## Deviation: S4 setup scoring (2026-10-22)

The reference listed `targets: []` for S4, and the scorer compared target lists. MLCA requires a non-empty target list for every supervised task, survival included (`inference.py` `validate()`), so no survival proposal could agree. All three S4 proposals chose problem type survival, duration `Overall Survival (Months)` and event `Overall Survival Status`, as declared. The scorer now compares duration and event for survival tasks (`scripts/score_inference.py`, test `tests/test_inference_score.py`). Re-scored files: `studies/s4_hnscc-00N/attempt-001/setup-agreement-rescored.json`; the original `setup-agreement.json` files are kept.

| | Original scoring | Corrected scoring |
|---|---|---|
| S4 setup agrees | 0/3 | 3/3 |
| Successful runs (completed, setup agrees, leakage clean) | 1/27 | 1/27 (unchanged; no S4 run completed) |
