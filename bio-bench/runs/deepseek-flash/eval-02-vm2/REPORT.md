# Evaluation batch 1

Model, provider, runtime source and rail setting fixed for every study in this batch.
Comparison plan: None
Acceptance outcome and rail effect are reported separately; failed runs are retained.

| Table / replicate | Completed | Setup agrees | Leakage: columns / groups / time | Success | Furthest event / step | Category | Repairs | Requests | Input / output / reasoning | Known cost USD | Wall s | Reported models |
|---|---|---|---|---|---|---|---:|---:|---|---:|---:|---|
| s2_proteome_pr / 1 | False | True | None / True / True | False | data / 02_data_audit_schema_079N0PAT | sandbox_denied | 2 | 7 | 244470 / 12734 / 10982 | None | 880.3955588340759 | {"deepseek-flash": 7} |
| s2_proteome_pr / 2 | False | True | None / True / True | False | data / 02_data_audit_schema_GDW3PVC4 | sandbox_denied | 2 | 7 | 244481 / 18078 / 16319 | None | 900.3220629692078 | {"deepseek-flash": 7} |
| s2_proteome_pr / 3 | False | True | True / True / True | False | interpret / 11_interpret_explain_model_0ZTQQHYB | runtime_error:RuntimeError | 3 | 36 | 329214 / 95739 / 78175 | None | 1624.2430186271667 | {"deepseek-flash": 36} |
| s2_proteome_her2 / 1 | True | True | True / True / True | True | report / 13_report_generate_reports_1WJDR2BD |  | 3 | 44 | 355948 / 92336 / 74855 | None | 1625.3916046619415 | {"deepseek-flash": 44} |
| s2_proteome_her2 / 2 | True | True | True / True / True | True | report / 12_report_generate_reports_115WK5NA |  | 1 | 39 | 576652 / 98870 / 83117 | None | 1730.0246987342834 | {"deepseek-flash": 39} |
| s2_proteome_her2 / 3 | True | True | True / True / True | True | report / 12_report_generate_reports_TYX1SESB |  | 1 | 40 | 344927 / 81678 / 67148 | None | 1580.6380212306976 | {"deepseek-flash": 40} |

## Environment preflight

Last check 2026-10-24T01:49:07.769355+00:00 on mlca-runner-2.us-central1-a.c.pelliscope-scout.internal, Python 3.13.17: default 3e677b448fb0 (matches reference), lifelines 53f3d70c0a00 (matches reference)

## Batch accounting

```json
{
  "requests": 173,
  "known": {
    "input_tokens": 2095692,
    "output_tokens": 399435,
    "reasoning_tokens": 330596,
    "cost_usd": null
  },
  "unknown": {
    "input_tokens": 0,
    "output_tokens": 0,
    "cost_usd": 173
  },
  "partial": true,
  "models": {
    "deepseek-flash": 173
  }
}
```

## Running accounting since batch-01

```json
{
  "requests": 355,
  "known": {
    "input_tokens": 2700108,
    "output_tokens": 879434,
    "reasoning_tokens": 737603,
    "cost_usd": null
  },
  "unknown": {
    "input_tokens": 21,
    "output_tokens": 21,
    "cost_usd": 355
  },
  "partial": true,
  "models": {
    "unavailable": 21,
    "deepseek-flash": 334
  }
}
```

Known totals are partial when unknown request counts are nonzero. Reasoning tokens are not added to completion tokens. All failed and interrupted attempts are retained. Individual model distributions and step IDs are in batch.json. Scorer outputs are external evidence, not agent work.
