# Development batch 10

Model, provider, runtime source and rail setting fixed for every study in this batch.
Comparison plan: None
Acceptance outcome and rail effect are reported separately; failed runs are retained.

| Table / replicate | Completed | Setup agrees | Leakage: columns / groups / time | Success | Furthest event / step | Category | Repairs | Requests | Input / output / reasoning | Known cost USD | Wall s | Reported models |
|---|---|---|---|---|---|---|---:|---:|---|---:|---:|---|
| heart_disease_cleveland / 1 | True | True | True / True / True | True | report / 12_report_generate_reports_3NJKW0VK |  | 0 | 37 | 95109 / 88610 / 73354 | None | 466.0480499267578 | {"deepseek-flash": 37} |
| bike_sharing_hourly / 1 | False | True | True / True / True | False | model / 07_model_train_baseline_EJMBYS73 | runtime_error:ValueError | 5 | 24 | 70657 / 126688 / 115086 | None | 625.0953786373138 | {"deepseek-flash": 24} |
| covertype / 1 | True | True | True / True / True | True | report / 14_report_generate_reports_V409W0N1 |  | 1 | 43 | 111099 / 101427 / 86061 | None | 2679.589192867279 | {"deepseek-flash": 43} |
| adult / 1 | True | True | True / True / True | True | report / 12_report_generate_reports_A944DQS5 |  | 2 | 38 | 103751 / 100832 / 84392 | None | 658.8715534210205 | {"deepseek-flash": 38} |
| dry_bean / 1 | True | True | True / True / True | True | report / 12_report_generate_reports_RA36R5KM |  | 1 | 39 | 104773 / 65883 / 52471 | None | 513.3998413085938 | {"deepseek-flash": 39} |
| seeds / 1 | True | False | True / True / True | False | report / 12_report_generate_reports_QWCCE6YK | setup_disagreement | 1 | 40 | 109844 / 77283 / 60636 | None | 416.9690799713135 | {"deepseek-flash": 40} |
| online_retail / 1 | True | True | True / True / True | True | report / 12_report_generate_reports_PZCY988Q |  | 0 | 37 | 95017 / 99523 / 84340 | None | 657.1673538684845 | {"deepseek-flash": 37} |

## Batch accounting

```json
{
  "requests": 258,
  "known": {
    "input_tokens": 690250,
    "output_tokens": 660246,
    "reasoning_tokens": 556340,
    "cost_usd": null
  },
  "unknown": {
    "input_tokens": 0,
    "output_tokens": 0,
    "cost_usd": 258
  },
  "partial": true,
  "models": {
    "deepseek-flash": 258
  }
}
```

## Running accounting since batch-01

```json
{
  "requests": 507,
  "known": {
    "input_tokens": 1377496,
    "output_tokens": 1308920,
    "reasoning_tokens": 1104118,
    "cost_usd": null
  },
  "unknown": {
    "input_tokens": 0,
    "output_tokens": 0,
    "cost_usd": 507
  },
  "partial": true,
  "models": {
    "deepseek-flash": 507
  }
}
```

Known totals are partial when unknown request counts are nonzero. Reasoning tokens are not added to completion tokens. All failed and interrupted attempts are retained. Individual model distributions and step IDs are in batch.json. Scorer outputs are external evidence, not agent work.
