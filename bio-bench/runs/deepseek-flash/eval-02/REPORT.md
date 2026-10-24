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
| s2_rna_her2 / 1 | False | None | None / None / None | False | setup / None | provider_error | 0 | 1 | None / None / None | None | 9.771857500076294 | {"unavailable": 1} |
| s2_rna_her2 / 2 | False | None | None / None / None | False | setup / None | provider_error | 0 | 1 | None / None / None | None | 9.824033737182617 | {"unavailable": 1} |
| s2_rna_her2 / 3 | False | None | None / None / None | False | setup / None | provider_error | 0 | 1 | None / None / None | None | 9.875542640686035 | {"unavailable": 1} |
| s2_proteome_er / 1 | False | None | None / None / None | False | setup / None | provider_error | 0 | 1 | None / None / None | None | 9.817577838897705 | {"unavailable": 1} |
| s2_proteome_er / 2 | False | None | None / None / None | False | setup / None | provider_error | 0 | 1 | None / None / None | None | 9.834676742553711 | {"unavailable": 1} |
| s2_proteome_er / 3 | True | True | True / True / True | True | report / 12_report_generate_reports_NM1C069N |  | 1 | 37 | 328606 / 85754 / 71218 | None | 1268.338407754898 | {"deepseek-flash": 37} |
| s2_proteome_pr / 1 | queued | None | None / None / None | None | setup / None |  | None | None | None / None / None | None | None | {} |
| s2_proteome_pr / 2 | queued | None | None / None / None | None | setup / None |  | None | None | None / None / None | None | None | {} |
| s2_proteome_pr / 3 | queued | None | None / None / None | None | setup / None |  | None | None | None / None / None | None | None | {} |
| s2_proteome_her2 / 1 | queued | None | None / None / None | None | setup / None |  | None | None | None / None / None | None | None | {} |
| s2_proteome_her2 / 2 | queued | None | None / None / None | None | setup / None |  | None | None | None / None / None | None | None | {} |
| s2_proteome_her2 / 3 | queued | None | None / None / None | None | setup / None |  | None | None | None / None / None | None | None | {} |
| s3_pam50 / 1 | queued | None | None / None / None | None | setup / None |  | None | None | None / None / None | None | None | {} |
| s3_pam50 / 2 | queued | None | None / None / None | None | setup / None |  | None | None | None / None / None | None | None | {} |
| s3_pam50 / 3 | queued | None | None / None / None | None | setup / None |  | None | None | None / None / None | None | None | {} |
| s4_hnscc / 1 | queued | None | None / None / None | None | setup / None |  | None | None | None / None / None | None | None | {} |
| s4_hnscc / 2 | queued | None | None / None / None | None | setup / None |  | None | None | None / None / None | None | None | {} |
| s4_hnscc / 3 | queued | None | None / None / None | None | setup / None |  | None | None | None / None / None | None | None | {} |

## Batch accounting

```json
{
  "requests": 386,
  "known": {
    "input_tokens": 4134922,
    "output_tokens": 890388,
    "reasoning_tokens": 742530,
    "cost_usd": null
  },
  "unknown": {
    "input_tokens": 7,
    "output_tokens": 7,
    "cost_usd": 386
  },
  "partial": true,
  "models": {
    "deepseek-flash": 379,
    "unavailable": 7
  }
}
```

## Running accounting since batch-01

```json
{
  "requests": 568,
  "known": {
    "input_tokens": 4739338,
    "output_tokens": 1370387,
    "reasoning_tokens": 1149537,
    "cost_usd": null
  },
  "unknown": {
    "input_tokens": 28,
    "output_tokens": 28,
    "cost_usd": 568
  },
  "partial": true,
  "models": {
    "unavailable": 28,
    "deepseek-flash": 540
  }
}
```

Known totals are partial when unknown request counts are nonzero. Reasoning tokens are not added to completion tokens. All failed and interrupted attempts are retained. Individual model distributions and step IDs are in batch.json. Scorer outputs are external evidence, not agent work.
