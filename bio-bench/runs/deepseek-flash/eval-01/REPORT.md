# Evaluation batch 1

Model, provider, runtime source and rail setting fixed for every study in this batch.
Comparison plan: None
Acceptance outcome and rail effect are reported separately; failed runs are retained.

| Table / replicate | Completed | Setup agrees | Leakage: columns / groups / time | Success | Furthest event / step | Category | Repairs | Requests | Input / output / reasoning | Known cost USD | Wall s | Reported models |
|---|---|---|---|---|---|---|---:|---:|---|---:|---:|---|
| s1_anemia / 1 | True | True | True / True / True | True | report / 12_report_generate_reports_MM39PEY8 |  | 1 | 37 | 95235 / 83659 / 70009 | None | 499.8124964237213 | {"deepseek-flash": 37} |
| s1_anemia / 2 | False | True | True / True / True | False | validate / 09_validate_score_validation_ZGS6S2F7 | gate_blocked | 0 | 26 | 60020 / 64551 / 54929 | None | 358.53750133514404 | {"deepseek-flash": 26} |
| s1_anemia / 3 | False | True | True / True / True | False | validate / 09_validate_score_validation_XEBNP8F1 | gate_blocked | 1 | 28 | 67966 / 57777 / 47637 | None | 331.1263391971588 | {"deepseek-flash": 28} |
| s2_rna_er / 1 | False | None | None / None / None | False | setup / None | provider_error | 0 | 1 | None / None / None | None | 45.05204510688782 | {"unavailable": 1} |
| s2_rna_er / 2 | False | None | None / None / None | False | setup / None | provider_error | 0 | 1 | None / None / None | None | 47.3443489074707 | {"unavailable": 1} |
| s2_rna_er / 3 | False | None | None / None / None | False | setup / None | provider_error | 0 | 1 | None / None / None | None | 45.94223356246948 | {"unavailable": 1} |
| s2_rna_pr / 1 | False | None | None / None / None | False | setup / None | provider_error | 0 | 1 | None / None / None | None | 43.91396713256836 | {"unavailable": 1} |
| s2_rna_pr / 2 | False | None | None / None / None | False | setup / None | provider_error | 0 | 1 | None / None / None | None | 45.35637092590332 | {"unavailable": 1} |
| s2_rna_pr / 3 | False | None | None / None / None | False | setup / None | provider_error | 0 | 1 | None / None / None | None | 49.838398694992065 | {"unavailable": 1} |
| s2_rna_her2 / 1 | False | None | None / None / None | False | setup / None | provider_error | 0 | 1 | None / None / None | None | 52.98206305503845 | {"unavailable": 1} |
| s2_rna_her2 / 2 | False | None | None / None / None | False | setup / None | provider_error | 0 | 1 | None / None / None | None | 49.44038462638855 | {"unavailable": 1} |
| s2_rna_her2 / 3 | False | None | None / None / None | False | setup / None | provider_error | 0 | 1 | None / None / None | None | 47.888829946517944 | {"unavailable": 1} |
| s2_proteome_er / 1 | False | None | None / None / None | False | setup / None | provider_error | 0 | 1 | None / None / None | None | 25.761230945587158 | {"unavailable": 1} |
| s2_proteome_er / 2 | False | None | None / None / None | False | setup / None | provider_error | 0 | 1 | None / None / None | None | 31.732096672058105 | {"unavailable": 1} |
| s2_proteome_er / 3 | False | None | None / None / None | False | setup / None | provider_error | 0 | 1 | None / None / None | None | 27.551349878311157 | {"unavailable": 1} |
| s2_proteome_pr / 1 | False | None | None / None / None | False | setup / None | provider_error | 0 | 1 | None / None / None | None | 28.284677505493164 | {"unavailable": 1} |
| s2_proteome_pr / 2 | False | None | None / None / None | False | setup / None | provider_error | 0 | 1 | None / None / None | None | 28.61809515953064 | {"unavailable": 1} |
| s2_proteome_pr / 3 | False | None | None / None / None | False | setup / None | provider_error | 0 | 1 | None / None / None | None | 29.22946834564209 | {"unavailable": 1} |
| s2_proteome_her2 / 1 | False | None | None / None / None | False | setup / None | provider_error | 0 | 1 | None / None / None | None | 29.580023288726807 | {"unavailable": 1} |
| s2_proteome_her2 / 2 | False | None | None / None / None | False | setup / None | provider_error | 0 | 1 | None / None / None | None | 30.32737898826599 | {"unavailable": 1} |
| s2_proteome_her2 / 3 | False | None | None / None / None | False | setup / None | provider_error | 0 | 1 | None / None / None | None | 30.936644315719604 | {"unavailable": 1} |
| s3_pam50 / 1 | False | None | None / None / None | False | setup / None | provider_error | 0 | 1 | None / None / None | None | 52.52911043167114 | {"unavailable": 1} |
| s3_pam50 / 2 | False | None | None / None / None | False | setup / None | provider_error | 0 | 1 | None / None / None | None | 47.76160168647766 | {"unavailable": 1} |
| s3_pam50 / 3 | False | None | None / None / None | False | setup / None | provider_error | 0 | 1 | None / None / None | None | 55.02856469154358 | {"unavailable": 1} |
| s4_hnscc / 1 | False | False | True / True / True | False | validate / 09_validate_score_validation_R2X2E0K4 | runtime_error:AttributeError | 5 | 30 | 151807 / 148315 / 128347 | None | 758.809808254242 | {"deepseek-flash": 30} |
| s4_hnscc / 2 | False | False | True / True / True | False | validate / 09_validate_score_validation_H25V5FH2 | runtime_error:AttributeError | 2 | 28 | 139767 / 113475 / 96919 | None | 624.77379322052 | {"deepseek-flash": 28} |
| s4_hnscc / 3 | False | False | None / True / True | False | preprocess / 04_split_freeze_random_RFYY7E43 | gate_blocked | 0 | 12 | 89621 / 12222 / 9166 | None | 118.18151521682739 | {"deepseek-flash": 12} |

## Batch accounting

```json
{
  "requests": 182,
  "known": {
    "input_tokens": 604416,
    "output_tokens": 479999,
    "reasoning_tokens": 407007,
    "cost_usd": null
  },
  "unknown": {
    "input_tokens": 21,
    "output_tokens": 21,
    "cost_usd": 182
  },
  "partial": true,
  "models": {
    "unavailable": 21,
    "deepseek-flash": 161
  }
}
```

## Running accounting since batch-01

```json
{
  "requests": 182,
  "known": {
    "input_tokens": 604416,
    "output_tokens": 479999,
    "reasoning_tokens": 407007,
    "cost_usd": null
  },
  "unknown": {
    "input_tokens": 21,
    "output_tokens": 21,
    "cost_usd": 182
  },
  "partial": true,
  "models": {
    "unavailable": 21,
    "deepseek-flash": 161
  }
}
```

Known totals are partial when unknown request counts are nonzero. Reasoning tokens are not added to completion tokens. All failed and interrupted attempts are retained. Individual model distributions and step IDs are in batch.json. Scorer outputs are external evidence, not agent work.
