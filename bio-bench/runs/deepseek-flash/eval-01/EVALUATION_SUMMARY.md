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
