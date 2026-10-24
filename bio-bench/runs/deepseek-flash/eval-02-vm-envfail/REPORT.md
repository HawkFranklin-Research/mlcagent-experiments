# Evaluation batch 1

Model, provider, runtime source and rail setting fixed for every study in this batch.
Comparison plan: None
Acceptance outcome and rail effect are reported separately; failed runs are retained.

| Table / replicate | Completed | Setup agrees | Leakage: columns / groups / time | Success | Furthest event / step | Category | Repairs | Requests | Input / output / reasoning | Known cost USD | Wall s | Reported models |
|---|---|---|---|---|---|---|---:|---:|---|---:|---:|---|
| s1_anemia / 1 | True | True | True / True / True | True | report / 12_report_generate_reports_EV4EA66K |  | 0 | 36 | 89767 / 81270 / 68795 | None | 488.8414270877838 | {"deepseek-flash": 36} |
| s1_anemia / 2 | True | True | True / True / True | True | report / 12_report_generate_reports_MBPCPXJM |  | 0 | 37 | 95180 / 75653 / 61639 | None | 462.1402485370636 | {"deepseek-flash": 37} |
| s1_anemia / 3 | True | True | True / True / True | True | report / 12_report_generate_reports_FJAT917S |  | 1 | 37 | 94564 / 102300 / 87311 | None | 600.1626541614532 | {"deepseek-flash": 37} |
| s2_rna_er / 1 | True | True | True / True / True | True | report / 12_report_generate_reports_N4BJFKYB |  | 0 | 36 | 500511 / 76118 / 61909 | None | 2642.944883584976 | {"deepseek-flash": 36} |
| s2_rna_er / 2 | True | True | True / True / True | True | report / 12_report_generate_reports_FKJTXZ13 |  | 0 | 39 | 512934 / 91049 / 76071 | None | 2773.0361003875732 | {"deepseek-flash": 39} |
| s2_rna_er / 3 | True | True | True / True / True | True | report / 12_report_generate_reports_GKJNPW3J |  | 3 | 39 | 517340 / 97080 / 80130 | None | 2867.8364205360413 | {"deepseek-flash": 39} |
| s2_rna_pr / 1 | True | True | True / True / True | True | report / 13_report_generate_reports_STRAYB70 |  | 1 | 43 | 535035 / 90060 / 74917 | None | 2871.5233538150787 | {"deepseek-flash": 43} |
| s2_rna_pr / 2 | True | True | True / True / True | True | report / 12_report_generate_reports_YJC2B2CX |  | 1 | 38 | 943470 / 99950 / 85009 | None | 2877.055286884308 | {"deepseek-flash": 38} |
| s2_rna_pr / 3 | True | True | True / True / True | True | report / 13_report_generate_reports_P26YHVFC |  | 0 | 39 | 517515 / 91154 / 75531 | None | 3486.0389189720154 | {"deepseek-flash": 37, "unavailable": 2} |
| s2_rna_her2 / 1 (studies/s2_rna_her2-001/attempt-001) | False | None | None / None / None | False | setup / None | provider_error | 0 | 1 | None / None / None | None | 9.771857500076294 | {"unavailable": 1} |
| s2_rna_her2 / 1 (attempt-002) | False | None | None / None / None | False | setup / None | runtime_error:MlcaUserError | 0 | 2 | 410490 / 535 / 261 | None | 132.53851079940796 | {"unavailable": 1, "deepseek-flash": 1} |
| s2_rna_her2 / 2 (studies/s2_rna_her2-002/attempt-001) | False | None | None / None / None | False | setup / None | provider_error | 0 | 1 | None / None / None | None | 9.824033737182617 | {"unavailable": 1} |
| s2_rna_her2 / 2 (attempt-002) | False | None | None / None / None | False | setup / None | runtime_error:MlcaUserError | 0 | 2 | 410490 / 1020 / 589 | None | 138.20729565620422 | {"unavailable": 1, "deepseek-flash": 1} |
| s2_rna_her2 / 3 (studies/s2_rna_her2-003/attempt-001) | False | None | None / None / None | False | setup / None | provider_error | 0 | 1 | None / None / None | None | 9.875542640686035 | {"unavailable": 1} |
| s2_rna_her2 / 3 (attempt-002) | False | None | None / None / None | False | setup / None | runtime_error:MlcaUserError | 0 | 2 | 410490 / 506 / 169 | None | 131.9820532798767 | {"unavailable": 1, "deepseek-flash": 1} |
| s2_proteome_er / 1 (studies/s2_proteome_er-001/attempt-001) | False | None | None / None / None | False | setup / None | provider_error | 0 | 1 | None / None / None | None | 9.817577838897705 | {"unavailable": 1} |
| s2_proteome_er / 1 (attempt-002) | False | None | None / None / None | False | setup / None | runtime_error:MlcaUserError | 0 | 2 | 234947 / 432 / 114 | None | 67.35215854644775 | {"unavailable": 1, "deepseek-flash": 1} |
| s2_proteome_er / 2 (studies/s2_proteome_er-002/attempt-001) | False | None | None / None / None | False | setup / None | provider_error | 0 | 1 | None / None / None | None | 9.834676742553711 | {"unavailable": 1} |
| s2_proteome_er / 2 (attempt-002) | False | None | None / None / None | False | setup / None | runtime_error:MlcaUserError | 0 | 2 | 234947 / 524 / 227 | None | 81.9174325466156 | {"unavailable": 1, "deepseek-flash": 1} |
| s2_proteome_er / 3 | True | True | True / True / True | True | report / 12_report_generate_reports_NM1C069N |  | 1 | 37 | 328606 / 85754 / 71218 | None | 1268.338407754898 | {"deepseek-flash": 37} |
| s2_proteome_pr / 1 | False | None | None / None / None | False | setup / None | runtime_error:MlcaUserError | 0 | 1 | 228438 / 434 / 161 | None | 68.41193342208862 | {"deepseek-flash": 1} |
| s2_proteome_pr / 2 | False | None | None / None / None | False | setup / None | runtime_error:MlcaUserError | 0 | 1 | 228438 / 268 / 44 | None | 62.69905686378479 | {"deepseek-flash": 1} |
| s2_proteome_pr / 3 | False | None | None / None / None | False | setup / None | runtime_error:MlcaUserError | 0 | 1 | 228438 / 431 / 152 | None | 62.87460899353027 | {"deepseek-flash": 1} |
| s2_proteome_her2 / 1 | False | None | None / None / None | False | setup / None | runtime_error:MlcaUserError | 0 | 1 | 238306 / 511 / 201 | None | 74.84054708480835 | {"deepseek-flash": 1} |
| s2_proteome_her2 / 2 | False | None | None / None / None | False | setup / None | runtime_error:MlcaUserError | 0 | 2 | 477060 / 881 / 176 | None | 75.24556374549866 | {"deepseek-flash": 2} |
| s2_proteome_her2 / 3 | False | None | None / None / None | False | setup / None | runtime_error:MlcaUserError | 0 | 1 | 238306 / 596 / 315 | None | 69.64797639846802 | {"deepseek-flash": 1} |
| s3_pam50 / 1 | False | None | None / None / None | False | setup / None | runtime_error:MlcaUserError | 0 | 1 | 68384 / 453 / 159 | None | 881.4658391475677 | {"deepseek-flash": 1} |
| s3_pam50 / 2 | False | None | None / None / None | False | setup / None | runtime_error:MlcaUserError | 0 | 2 | 137305 / 1259 / 427 | None | 910.6523911952972 | {"deepseek-flash": 2} |
| s3_pam50 / 3 | False | None | None / None / None | False | setup / None | runtime_error:MlcaUserError | 0 | 1 | 68384 / 554 / 159 | None | 946.912294626236 | {"deepseek-flash": 1} |
| s4_hnscc / 1 | False | None | None / None / None | False | setup / None | runtime_error:MlcaUserError | 0 | 2 | 18223 / 910 / 359 | None | 36.920363903045654 | {"deepseek-flash": 2} |
| s4_hnscc / 2 | False | None | None / None / None | False | setup / None | runtime_error:MlcaUserError | 0 | 1 | 8929 / 429 / 188 | None | 24.245160818099976 | {"deepseek-flash": 1} |
| s4_hnscc / 3 | False | None | None / None / None | False | setup / None | runtime_error:MlcaUserError | 0 | 1 | 8929 / 415 / 161 | None | 23.861490964889526 | {"deepseek-flash": 1} |

## Connection retries

The following jobs failed at the first setup request due to a connection error before any model response and were retried once:

| Job | Dataset | Replicate | Retried attempt | Reason | Retried at |
|---|---|---:|---|---|---|
| s2_rna_her2-001 | s2_rna_her2 | 1 | studies/s2_rna_her2-001/attempt-001 | connection_failure | 2026-10-23T10:03:07.259299+00:00 |
| s2_rna_her2-002 | s2_rna_her2 | 2 | studies/s2_rna_her2-002/attempt-001 | connection_failure | 2026-10-23T10:03:07.259313+00:00 |
| s2_rna_her2-003 | s2_rna_her2 | 3 | studies/s2_rna_her2-003/attempt-001 | connection_failure | 2026-10-23T10:03:07.259321+00:00 |
| s2_proteome_er-001 | s2_proteome_er | 1 | studies/s2_proteome_er-001/attempt-001 | connection_failure | 2026-10-23T10:03:07.259326+00:00 |
| s2_proteome_er-002 | s2_proteome_er | 2 | studies/s2_proteome_er-002/attempt-001 | connection_failure | 2026-10-23T10:03:07.259331+00:00 |

## Batch accounting

```json
{
  "requests": 406,
  "known": {
    "input_tokens": 7785426,
    "output_tokens": 900546,
    "reasoning_tokens": 746392,
    "cost_usd": null
  },
  "unknown": {
    "input_tokens": 7,
    "output_tokens": 7,
    "cost_usd": 406
  },
  "partial": true,
  "models": {
    "deepseek-flash": 399,
    "unavailable": 7
  }
}
```

## Running accounting since batch-01

```json
{
  "requests": 588,
  "known": {
    "input_tokens": 8389842,
    "output_tokens": 1380545,
    "reasoning_tokens": 1153399,
    "cost_usd": null
  },
  "unknown": {
    "input_tokens": 28,
    "output_tokens": 28,
    "cost_usd": 588
  },
  "partial": true,
  "models": {
    "unavailable": 28,
    "deepseek-flash": 560
  }
}
```

Known totals are partial when unknown request counts are nonzero. Reasoning tokens are not added to completion tokens. All failed and interrupted attempts are retained. Individual model distributions and step IDs are in batch.json. Scorer outputs are external evidence, not agent work.
