# Evaluation batch 1

Model, provider, runtime source and rail setting fixed for every study in this batch.
Comparison plan: None
Acceptance outcome and rail effect are reported separately; failed runs are retained.

| Table / replicate | Completed | Setup agrees | Leakage: columns / groups / time | Success | Furthest event / step | Category | Repairs | Requests | Input / output / reasoning | Known cost USD | Wall s | Reported models |
|---|---|---|---|---|---|---|---:|---:|---|---:|---:|---|
| s3_pam50 / 1 (studies/s3_pam50-001/attempt-001) | False | True | None / True / True | False | data / 02_data_audit_schema_2AWVWD51 | sandbox_denied | 2 | 8 | 152977 / 9784 / 7293 | None | 1101.496365070343 | {"deepseek-flash": 8} |
| s3_pam50 / 1 (studies/s3_pam50-001/attempt-002) | False | None | None / None / None | False | setup / None | interrupted_by_author | 2 | 8 | 152977 / 9784 / 7293 | None | 13.558547258377075 | {"deepseek-flash": 8} |
| s3_pam50 / 1 (attempt-003) | True | True | True / True / True | True | report / 12_report_generate_reports_Q6E0Z0X5 |  | 3 | 46 | 317349 / 86193 / 69775 | None | 2845.06978058815 | {"deepseek-flash": 46} |
| s3_pam50 / 2 (studies/s3_pam50-002/attempt-001) | False | True | None / True / True | False | data / 02_data_count_target_3PARQZYR | sandbox_denied | 2 | 7 | 83955 / 13729 / 11514 | None | 1138.823141336441 | {"deepseek-flash": 7} |
| s3_pam50 / 2 (studies/s3_pam50-002/attempt-002) | False | None | None / None / None | False | setup / None | interrupted_by_author | 2 | 7 | 83955 / 13729 / 11514 | None | 12.438961267471313 | {"deepseek-flash": 7} |
| s3_pam50 / 2 (attempt-003) | True | True | True / True / True | True | report / 12_report_generate_reports_RP3FS275 |  | 2 | 43 | 239319 / 84097 / 68231 | None | 2874.63072514534 | {"deepseek-flash": 43} |
| s3_pam50 / 3 | True | True | True / True / True | True | report / 13_report_generate_reports_C0XKAAXT |  | 0 | 41 | 168092 / 86255 / 72321 | None | 2789.6756908893585 | {"deepseek-flash": 41} |
| s4_hnscc / 1 (studies/s4_hnscc-001/attempt-001) | False | False | None / True / True | False | data / 02_data_count_target_VRDH8CQK | sandbox_denied | 2 | 7 | 26124 / 24725 / 20700 | None | 156.3315155506134 | {"deepseek-flash": 7} |
| s4_hnscc / 1 (attempt-002) | False | True | True / True / True | False | validate / 09_validate_score_validation_DHA69FVT | runtime_error:AttributeError | 4 | 34 | 103226 / 126776 / 107169 | None | 659.5457627773285 | {"deepseek-flash": 34} |
| s4_hnscc / 2 | False | True | True / True / True | False | model / 09_model_train_alternative_Z1SYV3DJ | runtime_error:KeyError | 7 | 32 | 96753 / 131934 / 114794 | None | 910.688898563385 | {"deepseek-flash": 32} |
| s4_hnscc / 3 | False | True | True / True / True | False | validate / 09_validate_score_validation_FHRT9SW2 | runtime_error:EmptyDataError | 2 | 29 | 89461 / 95226 / 81301 | None | 933.9413912296295 | {"deepseek-flash": 29} |

## Retried attempts

These attempts ended for infrastructure reasons (connection or environment failure at setup, machine shutdown or loss, operator interrupt) and were run again as new attempts; the earlier attempts are kept and listed above:

| Job | Dataset | Replicate | Retried attempt | Reason | Retried at |
|---|---|---:|---|---|---|
| s3_pam50-001 | s3_pam50 | 1 | studies/s3_pam50-001/attempt-001 | sandbox_setup_failure | 2026-10-24T06:05:11.227131+00:00 |
| s3_pam50-001 | s3_pam50 | 1 | studies/s3_pam50-001/attempt-002 | interrupted_by_author | 2026-10-24T08:14:57.591337+00:00 |
| s3_pam50-002 | s3_pam50 | 2 | studies/s3_pam50-002/attempt-001 | sandbox_setup_failure | 2026-10-24T06:05:11.227375+00:00 |
| s3_pam50-002 | s3_pam50 | 2 | studies/s3_pam50-002/attempt-002 | interrupted_by_author | 2026-10-24T08:14:57.591347+00:00 |
| s4_hnscc-001 | s4_hnscc | 1 | studies/s4_hnscc-001/attempt-001 | sandbox_setup_failure | 2026-10-24T06:05:11.227560+00:00 |

## Environment preflight

Last check 2026-10-24T08:14:57.586935+00:00 on mlca-runner-3.us-central1-a.c.pelliscope-scout.internal, Python 3.13.17: default 3e677b448fb0 (matches reference), lifelines 53f3d70c0a00 (matches reference)

## Batch accounting

```json
{
  "requests": 225,
  "known": {
    "input_tokens": 1014200,
    "output_tokens": 610481,
    "reasoning_tokens": 513591,
    "cost_usd": null
  },
  "unknown": {
    "input_tokens": 0,
    "output_tokens": 0,
    "cost_usd": 225
  },
  "partial": true,
  "models": {
    "deepseek-flash": 225
  }
}
```

## Running accounting since batch-01

```json
{
  "requests": 774,
  "known": {
    "input_tokens": 5911508,
    "output_tokens": 1940356,
    "reasoning_tokens": 1627077,
    "cost_usd": null
  },
  "unknown": {
    "input_tokens": 21,
    "output_tokens": 21,
    "cost_usd": 774
  },
  "partial": true,
  "models": {
    "unavailable": 21,
    "deepseek-flash": 753
  }
}
```

Known totals are partial when unknown request counts are nonzero. Reasoning tokens are not added to completion tokens. All failed and interrupted attempts are retained. Individual model distributions and step IDs are in batch.json. Scorer outputs are external evidence, not agent work.
