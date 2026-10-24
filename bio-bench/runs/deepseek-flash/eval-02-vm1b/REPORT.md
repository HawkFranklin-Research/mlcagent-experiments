# Evaluation batch 1

Model, provider, runtime source and rail setting fixed for every study in this batch.
Comparison plan: None
Acceptance outcome and rail effect are reported separately; failed runs are retained.

| Table / replicate | Completed | Setup agrees | Leakage: columns / groups / time | Success | Furthest event / step | Category | Repairs | Requests | Input / output / reasoning | Known cost USD | Wall s | Reported models |
|---|---|---|---|---|---|---|---:|---:|---|---:|---:|---|
| s2_proteome_er / 1 | True | True | True / True / True | True | report / 12_report_generate_reports_DRZBM2T9 |  | 2 | 40 | 340588 / 92116 / 76587 | None | 2027.6657195091248 | {"deepseek-flash": 40} |
| s2_proteome_er / 2 | True | True | True / True / True | True | report / 12_report_generate_reports_6DD55Z9B |  | 1 | 38 | 331671 / 70578 / 57258 | None | 1617.1077890396118 | {"deepseek-flash": 38} |

## Environment preflight

Last check 2026-10-24T04:05:13.364597+00:00 on mlca-eval-runner.us-central1-a.c.pelliscope-scout.internal, Python 3.13.17: default 3e677b448fb0 (matches reference), lifelines 53f3d70c0a00 (matches reference)

## Batch accounting

```json
{
  "requests": 78,
  "known": {
    "input_tokens": 672259,
    "output_tokens": 162694,
    "reasoning_tokens": 133845,
    "cost_usd": null
  },
  "unknown": {
    "input_tokens": 0,
    "output_tokens": 0,
    "cost_usd": 78
  },
  "partial": true,
  "models": {
    "deepseek-flash": 78
  }
}
```

## Running accounting since batch-01

```json
{
  "requests": 376,
  "known": {
    "input_tokens": 2801616,
    "output_tokens": 930440,
    "reasoning_tokens": 782890,
    "cost_usd": null
  },
  "unknown": {
    "input_tokens": 21,
    "output_tokens": 21,
    "cost_usd": 376
  },
  "partial": true,
  "models": {
    "unavailable": 21,
    "deepseek-flash": 355
  }
}
```

Known totals are partial when unknown request counts are nonzero. Reasoning tokens are not added to completion tokens. All failed and interrupted attempts are retained. Individual model distributions and step IDs are in batch.json. Scorer outputs are external evidence, not agent work.
