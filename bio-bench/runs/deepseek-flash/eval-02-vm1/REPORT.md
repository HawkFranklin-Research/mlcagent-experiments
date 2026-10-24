# Evaluation batch 1

Model, provider, runtime source and rail setting fixed for every study in this batch.
Comparison plan: None
Acceptance outcome and rail effect are reported separately; failed runs are retained.

| Table / replicate | Completed | Setup agrees | Leakage: columns / groups / time | Success | Furthest event / step | Category | Repairs | Requests | Input / output / reasoning | Known cost USD | Wall s | Reported models |
|---|---|---|---|---|---|---|---:|---:|---|---:|---:|---|
| s2_rna_her2 / 1 | True | True | True / True / True | True | report / 12_report_generate_reports_PR7VWJ8R |  | 1 | 39 | 512746 / 77117 / 62067 | None | 4113.382006406784 | {"deepseek-flash": 39} |
| s2_rna_her2 / 2 | True | True | True / True / True | True | report / 12_report_generate_reports_NZGXE8P0 |  | 0 | 36 | 499954 / 102821 / 87472 | None | 4257.873511791229 | {"deepseek-flash": 36} |
| s2_rna_her2 / 3 | True | True | True / True / True | True | report / 13_report_generate_reports_XM5RMXZM |  | 0 | 41 | 512241 / 107809 / 92499 | None | 4052.591289281845 | {"deepseek-flash": 41} |

## Environment preflight

Last check 2026-10-24T01:48:56.217373+00:00 on mlca-eval-runner.us-central1-a.c.pelliscope-scout.internal, Python 3.13.17: default 3e677b448fb0 (matches reference), lifelines 53f3d70c0a00 (matches reference)

## Batch accounting

```json
{
  "requests": 116,
  "known": {
    "input_tokens": 1524941,
    "output_tokens": 287747,
    "reasoning_tokens": 242038,
    "cost_usd": null
  },
  "unknown": {
    "input_tokens": 0,
    "output_tokens": 0,
    "cost_usd": 116
  },
  "partial": true,
  "models": {
    "deepseek-flash": 116
  }
}
```

## Running accounting since batch-01

```json
{
  "requests": 298,
  "known": {
    "input_tokens": 2129357,
    "output_tokens": 767746,
    "reasoning_tokens": 649045,
    "cost_usd": null
  },
  "unknown": {
    "input_tokens": 21,
    "output_tokens": 21,
    "cost_usd": 298
  },
  "partial": true,
  "models": {
    "unavailable": 21,
    "deepseek-flash": 277
  }
}
```

Known totals are partial when unknown request counts are nonzero. Reasoning tokens are not added to completion tokens. All failed and interrupted attempts are retained. Individual model distributions and step IDs are in batch.json. Scorer outputs are external evidence, not agent work.
