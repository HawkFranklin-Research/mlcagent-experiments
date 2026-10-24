# Evaluation batch 1

Model, provider, runtime source and rail setting fixed for every study in this batch.
Comparison plan: None
Acceptance outcome and rail effect are reported separately; failed runs are retained.

| Table / replicate | Completed | Setup agrees | Leakage: columns / groups / time | Success | Furthest event / step | Category | Repairs | Requests | Input / output / reasoning | Known cost USD | Wall s | Reported models |
|---|---|---|---|---|---|---|---:|---:|---|---:|---:|---|
| s3_pam50 / 1 | False | True | None / True / True | False | data / 02_data_audit_schema_2AWVWD51 | sandbox_denied | 2 | 8 | 152977 / 9784 / 7293 | None | 1101.496365070343 | {"deepseek-flash": 8} |
| s3_pam50 / 2 | False | True | None / True / True | False | data / 02_data_count_target_3PARQZYR | sandbox_denied | 2 | 7 | 83955 / 13729 / 11514 | None | 1138.823141336441 | {"deepseek-flash": 7} |
| s3_pam50 / 3 | True | True | True / True / True | True | report / 13_report_generate_reports_C0XKAAXT |  | 0 | 41 | 168092 / 86255 / 72321 | None | 2789.6756908893585 | {"deepseek-flash": 41} |
| s4_hnscc / 1 | False | False | None / True / True | False | data / 02_data_count_target_VRDH8CQK | sandbox_denied | 2 | 7 | 26124 / 24725 / 20700 | None | 156.3315155506134 | {"deepseek-flash": 7} |
| s4_hnscc / 2 | False | True | True / True / True | False | model / 09_model_train_alternative_Z1SYV3DJ | runtime_error:KeyError | 7 | 32 | 96753 / 131934 / 114794 | None | 910.688898563385 | {"deepseek-flash": 32} |
| s4_hnscc / 3 | False | True | True / True / True | False | validate / 09_validate_score_validation_FHRT9SW2 | runtime_error:EmptyDataError | 2 | 29 | 89461 / 95226 / 81301 | None | 933.9413912296295 | {"deepseek-flash": 29} |

## Environment preflight

Last check 2026-10-24T01:49:20.244270+00:00 on mlca-runner-3.us-central1-a.c.pelliscope-scout.internal, Python 3.13.17: default 3e677b448fb0 (matches reference), lifelines 53f3d70c0a00 (matches reference)

## Batch accounting

```json
{
  "requests": 124,
  "known": {
    "input_tokens": 617362,
    "output_tokens": 361653,
    "reasoning_tokens": 307923,
    "cost_usd": null
  },
  "unknown": {
    "input_tokens": 0,
    "output_tokens": 0,
    "cost_usd": 124
  },
  "partial": true,
  "models": {
    "deepseek-flash": 124
  }
}
```

## Running accounting since batch-01

```json
{
  "requests": 306,
  "known": {
    "input_tokens": 1221778,
    "output_tokens": 841652,
    "reasoning_tokens": 714930,
    "cost_usd": null
  },
  "unknown": {
    "input_tokens": 21,
    "output_tokens": 21,
    "cost_usd": 306
  },
  "partial": true,
  "models": {
    "unavailable": 21,
    "deepseek-flash": 285
  }
}
```

Known totals are partial when unknown request counts are nonzero. Reasoning tokens are not added to completion tokens. All failed and interrupted attempts are retained. Individual model distributions and step IDs are in batch.json. Scorer outputs are external evidence, not agent work.
