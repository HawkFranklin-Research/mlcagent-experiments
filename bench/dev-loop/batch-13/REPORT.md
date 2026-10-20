# Development batch 13

Model, provider, runtime source and rail setting fixed for every study in this batch.
Comparison plan: None
Acceptance outcome and rail effect are reported separately; failed runs are retained.

| Table / replicate | Completed | Setup agrees | Leakage: columns / groups / time | Success | Furthest event / step | Category | Repairs | Requests | Input / output / reasoning | Known cost USD | Wall s | Reported models |
|---|---|---|---|---|---|---|---:|---:|---|---:|---:|---|
| breast_cancer_diagnostic / 1 | True | True | True / True / True | True | report / 12_report_generate_reports_DZTWZ8A2 |  | 0 | 36 | 92706 / 57154 / 44739 | None | 350.8218502998352 | {"deepseek-flash": 36} |
| abalone / 1 | True | True | True / True / True | True | report / 12_report_generate_reports_EYTE6QA1 |  | 5 | 41 | 116403 / 126260 / 107313 | None | 640.4088189601898 | {"deepseek-flash": 41} |

## Batch accounting

```json
{
  "requests": 77,
  "known": {
    "input_tokens": 209109,
    "output_tokens": 183414,
    "reasoning_tokens": 152052,
    "cost_usd": null
  },
  "unknown": {
    "input_tokens": 0,
    "output_tokens": 0,
    "cost_usd": 77
  },
  "partial": true,
  "models": {
    "deepseek-flash": 77
  }
}
```

## Running accounting since batch-01

```json
{
  "requests": 836,
  "known": {
    "input_tokens": 2247315,
    "output_tokens": 2080949,
    "reasoning_tokens": 1749426,
    "cost_usd": null
  },
  "unknown": {
    "input_tokens": 0,
    "output_tokens": 0,
    "cost_usd": 836
  },
  "partial": true,
  "models": {
    "deepseek-flash": 836
  }
}
```

Known totals are partial when unknown request counts are nonzero. Reasoning tokens are not added to completion tokens. All failed and interrupted attempts are retained. Individual model distributions and step IDs are in batch.json. Scorer outputs are external evidence, not agent work.
