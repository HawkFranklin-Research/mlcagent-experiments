# Development batch 11

Model, provider, runtime source and rail setting fixed for every study in this batch.
Comparison plan: None
Acceptance outcome and rail effect are reported separately; failed runs are retained.

| Table / replicate | Completed | Setup agrees | Leakage: columns / groups / time | Success | Furthest event / step | Category | Repairs | Requests | Input / output / reasoning | Known cost USD | Wall s | Reported models |
|---|---|---|---|---|---|---|---:|---:|---|---:|---:|---|
| heart_disease_cleveland / 1 | False | True | True / True / True | False | validate / 09_validate_score_validation_9YKZ0BNR | gate_blocked | 1 | 27 | 65375 / 43833 / 34906 | None | 255.95670580863953 | {"deepseek-flash": 27} |
| heart_disease_cleveland / 2 | True | True | True / True / True | True | report / 12_report_generate_reports_MDVZN2BB |  | 1 | 37 | 95978 / 74634 / 61486 | None | 419.17844319343567 | {"deepseek-flash": 37} |
| covertype / 1 | True | True | True / True / True | True | report / 12_report_generate_reports_J61TXTZA |  | 0 | 38 | 102520 / 70219 / 57606 | None | 2535.146949529648 | {"deepseek-flash": 38} |
| covertype / 2 | True | True | True / True / True | True | report / 12_report_generate_reports_60059HZ8 |  | 4 | 40 | 113753 / 112671 / 95194 | None | 3005.4119262695312 | {"deepseek-flash": 40} |

## Batch accounting

```json
{
  "requests": 142,
  "known": {
    "input_tokens": 377626,
    "output_tokens": 301357,
    "reasoning_tokens": 249192,
    "cost_usd": null
  },
  "unknown": {
    "input_tokens": 0,
    "output_tokens": 0,
    "cost_usd": 142
  },
  "partial": true,
  "models": {
    "deepseek-flash": 142
  }
}
```

## Running accounting since batch-01

```json
{
  "requests": 649,
  "known": {
    "input_tokens": 1755122,
    "output_tokens": 1610277,
    "reasoning_tokens": 1353310,
    "cost_usd": null
  },
  "unknown": {
    "input_tokens": 0,
    "output_tokens": 0,
    "cost_usd": 649
  },
  "partial": true,
  "models": {
    "deepseek-flash": 649
  }
}
```

Known totals are partial when unknown request counts are nonzero. Reasoning tokens are not added to completion tokens. All failed and interrupted attempts are retained. Individual model distributions and step IDs are in batch.json. Scorer outputs are external evidence, not agent work.
