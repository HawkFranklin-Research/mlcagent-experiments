# Development batch 12

Model, provider, runtime source and rail setting fixed for every study in this batch.
Comparison plan: None
Acceptance outcome and rail effect are reported separately; failed runs are retained.

| Table / replicate | Completed | Setup agrees | Leakage: columns / groups / time | Success | Furthest event / step | Category | Repairs | Requests | Input / output / reasoning | Known cost USD | Wall s | Reported models |
|---|---|---|---|---|---|---|---:|---:|---|---:|---:|---|
| bike_sharing_hourly / 1 | True | True | True / True / True | True | report / 12_report_generate_reports_5TWB2RMT |  | 0 | 36 | 90185 / 100735 / 87506 | None | 569.8737096786499 | {"deepseek-flash": 36} |
| bike_sharing_hourly / 2 | True | True | True / True / True | True | report / 12_report_generate_reports_JMQ69ZC4 |  | 2 | 38 | 102456 / 112327 / 95853 | None | 595.0262675285339 | {"deepseek-flash": 38} |
| bike_sharing_hourly / 3 | True | False | True / True / True | False | report / 12_report_generate_reports_J621JA27 | setup_disagreement | 0 | 36 | 90443 / 74196 / 60705 | None | 416.75778579711914 | {"deepseek-flash": 36} |

## Batch accounting

```json
{
  "requests": 110,
  "known": {
    "input_tokens": 283084,
    "output_tokens": 287258,
    "reasoning_tokens": 244064,
    "cost_usd": null
  },
  "unknown": {
    "input_tokens": 0,
    "output_tokens": 0,
    "cost_usd": 110
  },
  "partial": true,
  "models": {
    "deepseek-flash": 110
  }
}
```

## Running accounting since batch-01

```json
{
  "requests": 759,
  "known": {
    "input_tokens": 2038206,
    "output_tokens": 1897535,
    "reasoning_tokens": 1597374,
    "cost_usd": null
  },
  "unknown": {
    "input_tokens": 0,
    "output_tokens": 0,
    "cost_usd": 759
  },
  "partial": true,
  "models": {
    "deepseek-flash": 759
  }
}
```

Known totals are partial when unknown request counts are nonzero. Reasoning tokens are not added to completion tokens. All failed and interrupted attempts are retained. Individual model distributions and step IDs are in batch.json. Scorer outputs are external evidence, not agent work.
