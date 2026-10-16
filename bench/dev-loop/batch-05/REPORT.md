# Development batch 5

Development only; model, runtime source and rails fixed for this batch. No comparison with exploratory rounds.

| Table / replicate | Completed | Setup agrees | Leakage: columns / groups / time | Success | Furthest event / step | Category | Repairs | Requests | Input / output / reasoning | Known cost USD | Wall s | Reported models |
|---|---|---|---|---|---|---|---:|---:|---|---:|---:|---|
| bike_sharing_hourly / 1 | True | True | True / True / True | True | report / 12_report_generate_reports_P945X8EC |  | 2 | 39 | 83860 / 76369 / 64598 | 0.092760514 | 541.3424990177155 | {"deepseek/deepseek-v4.1-flash": 39} |
| optical_digits / 1 | False | True | True / True / True | False | validate / 09_validate_score_validation_35RG4J23 | runtime_error:ValueError | 7 | 34 | 96782 / 116936 / 103709 | 0.132198635 | 680.75146484375 | {"deepseek/deepseek-v4.1-flash": 34} |
| online_retail / 1 | False | False | None / None / True | False | split / 04_split_freeze_group_NS5AQEY4 | runtime_error:MlcaUserError | 0 | 11 | 20167 / 22528 / 18344 | 0.0192847968 | 377.7438020706177 | {"deepseek/deepseek-v4.1-flash": 11} |

## Batch accounting

```json
{
  "requests": 84,
  "known": {
    "input_tokens": 200809,
    "output_tokens": 215833,
    "reasoning_tokens": 186651,
    "cost_usd": 0.2442439458
  },
  "unknown": {
    "input_tokens": 0,
    "output_tokens": 0,
    "cost_usd": 0
  },
  "partial": false,
  "models": {
    "deepseek/deepseek-v4.1-flash": 84
  }
}
```

## Running accounting since batch-01

```json
{
  "requests": 1082,
  "known": {
    "input_tokens": 2470126,
    "output_tokens": 4343680,
    "reasoning_tokens": 4014228,
    "cost_usd": 4.3661140112
  },
  "unknown": {
    "input_tokens": 0,
    "output_tokens": 0,
    "cost_usd": 0
  },
  "partial": false,
  "models": {
    "deepseek/deepseek-v4.1-flash": 1082
  }
}
```

Known totals are partial when unknown request counts are nonzero. Reasoning tokens are not added to completion tokens. All failed and interrupted attempts are retained. Individual model distributions and step IDs are in batch.json. Scorer outputs are external evidence, not agent work.
