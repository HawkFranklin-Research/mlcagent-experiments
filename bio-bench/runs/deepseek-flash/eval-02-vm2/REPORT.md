# Evaluation batch 1

Model, provider, runtime source and rail setting fixed for every study in this batch.
Comparison plan: None
Acceptance outcome and rail effect are reported separately; failed runs are retained.

| Table / replicate | Completed | Setup agrees | Leakage: columns / groups / time | Success | Furthest event / step | Category | Repairs | Requests | Input / output / reasoning | Known cost USD | Wall s | Reported models |
|---|---|---|---|---|---|---|---:|---:|---|---:|---:|---|
| s2_proteome_pr / 1 (studies/s2_proteome_pr-001/attempt-001) | False | True | None / True / True | False | data / 02_data_audit_schema_079N0PAT | sandbox_denied | 2 | 7 | 244470 / 12734 / 10982 | None | 880.3955588340759 | {"deepseek-flash": 7} |
| s2_proteome_pr / 1 (studies/s2_proteome_pr-001/attempt-002) | False | None | None / None / None | False | setup / None | interrupted_by_author | 2 | 7 | 244470 / 12734 / 10982 | None | 10.35576319694519 | {"deepseek-flash": 7} |
| s2_proteome_pr / 1 (attempt-003) | True | True | True / True / True | True | report / 12_report_generate_reports_65A2M82B |  | 2 | 44 | 566275 / 107898 / 89975 | None | 1448.8041410446167 | {"deepseek-flash": 44} |
| s2_proteome_pr / 2 (studies/s2_proteome_pr-002/attempt-001) | False | True | None / True / True | False | data / 02_data_audit_schema_GDW3PVC4 | sandbox_denied | 2 | 7 | 244481 / 18078 / 16319 | None | 900.3220629692078 | {"deepseek-flash": 7} |
| s2_proteome_pr / 2 (studies/s2_proteome_pr-002/attempt-002) | False | None | None / None / None | False | setup / None | interrupted_by_author | 2 | 7 | 244481 / 18078 / 16319 | None | 10.360177755355835 | {"deepseek-flash": 7} |
| s2_proteome_pr / 2 (attempt-003) | False | None | None / None / None | False | setup / None | invalid_inference | 2 | 9 | 701878 / 18989 / 16393 | None | 21.959044933319092 | {"deepseek-flash": 9} |
| s2_proteome_pr / 3 | False | True | True / True / True | False | interpret / 11_interpret_explain_model_0ZTQQHYB | runtime_error:RuntimeError | 3 | 36 | 329214 / 95739 / 78175 | None | 1624.2430186271667 | {"deepseek-flash": 36} |
| s2_proteome_her2 / 1 | True | True | True / True / True | True | report / 13_report_generate_reports_1WJDR2BD |  | 3 | 44 | 355948 / 92336 / 74855 | None | 1625.3916046619415 | {"deepseek-flash": 44} |
| s2_proteome_her2 / 2 | True | True | True / True / True | True | report / 12_report_generate_reports_115WK5NA |  | 1 | 39 | 576652 / 98870 / 83117 | None | 1730.0246987342834 | {"deepseek-flash": 39} |
| s2_proteome_her2 / 3 | True | True | True / True / True | True | report / 12_report_generate_reports_TYX1SESB |  | 1 | 40 | 344927 / 81678 / 67148 | None | 1580.6380212306976 | {"deepseek-flash": 40} |

## Retried attempts

These attempts ended for infrastructure reasons (connection or environment failure at setup, machine shutdown or loss, operator interrupt) and were run again as new attempts; the earlier attempts are kept and listed above:

| Job | Dataset | Replicate | Retried attempt | Reason | Retried at |
|---|---|---:|---|---|---|
| s2_proteome_pr-001 | s2_proteome_pr | 1 | studies/s2_proteome_pr-001/attempt-001 | sandbox_setup_failure | 2026-10-24T06:04:47.849946+00:00 |
| s2_proteome_pr-001 | s2_proteome_pr | 1 | studies/s2_proteome_pr-001/attempt-002 | interrupted_by_author | 2026-10-24T08:14:31.798234+00:00 |
| s2_proteome_pr-002 | s2_proteome_pr | 2 | studies/s2_proteome_pr-002/attempt-001 | sandbox_setup_failure | 2026-10-24T06:04:47.850115+00:00 |
| s2_proteome_pr-002 | s2_proteome_pr | 2 | studies/s2_proteome_pr-002/attempt-002 | interrupted_by_author | 2026-10-24T08:14:31.798245+00:00 |

## Environment preflight

Last check 2026-10-24T08:14:31.793911+00:00 on mlca-runner-2.us-central1-a.c.pelliscope-scout.internal, Python 3.13.17: default 3e677b448fb0 (matches reference), lifelines 53f3d70c0a00 (matches reference)

## Batch accounting

```json
{
  "requests": 212,
  "known": {
    "input_tokens": 2874894,
    "output_tokens": 495510,
    "reasoning_tokens": 409663,
    "cost_usd": null
  },
  "unknown": {
    "input_tokens": 0,
    "output_tokens": 0,
    "cost_usd": 212
  },
  "partial": true,
  "models": {
    "deepseek-flash": 212
  }
}
```

## Running accounting since batch-01

```json
{
  "requests": 712,
  "known": {
    "input_tokens": 6293872,
    "output_tokens": 1787603,
    "reasoning_tokens": 1500476,
    "cost_usd": null
  },
  "unknown": {
    "input_tokens": 21,
    "output_tokens": 21,
    "cost_usd": 712
  },
  "partial": true,
  "models": {
    "unavailable": 21,
    "deepseek-flash": 691
  }
}
```

Known totals are partial when unknown request counts are nonzero. Reasoning tokens are not added to completion tokens. All failed and interrupted attempts are retained. Individual model distributions and step IDs are in batch.json. Scorer outputs are external evidence, not agent work.
