# Development batch 7

Development only; model, runtime source and rails fixed for this batch. No comparison with exploratory rounds.

| Table / replicate | Completed | Setup agrees | Leakage: columns / groups / time | Success | Furthest event / step | Category | Repairs | Requests | Input / output / reasoning | Known cost USD | Wall s | Reported models |
|---|---|---|---|---|---|---|---:|---:|---|---:|---:|---|
| optical_digits / 1 | True | True | True / True / True | True | report / 12_report_generate_reports_XNF6QB5E |  | 1 | 42 | 124692 / 64589 / 51026 | 0.074836174 | 493.57321786880493 | {"deepseek/deepseek-v4.1-flash": 42} |
| online_retail / 1 | True | False | True / True / True | False | report / 12_report_generate_reports_12HST9Y3 | setup_disagreement | 8 | 45 | 126015 / 137994 / 122437 | 0.144987094624 | 1048.6862478256226 | {"deepseek/deepseek-v4.1-flash": 45} |

## Batch accounting

```json
{
  "requests": 87,
  "known": {
    "input_tokens": 250707,
    "output_tokens": 202583,
    "reasoning_tokens": 173463,
    "cost_usd": 0.219823268624
  },
  "unknown": {
    "input_tokens": 0,
    "output_tokens": 0,
    "cost_usd": 0
  },
  "partial": false,
  "models": {
    "deepseek/deepseek-v4.1-flash": 87
  }
}
```

## Running accounting since batch-01

```json
{
  "requests": 1214,
  "known": {
    "input_tokens": 2850962,
    "output_tokens": 4649765,
    "reasoning_tokens": 4276091,
    "cost_usd": 4.701877847824
  },
  "unknown": {
    "input_tokens": 0,
    "output_tokens": 0,
    "cost_usd": 0
  },
  "partial": false,
  "models": {
    "deepseek/deepseek-v4.1-flash": 1214
  }
}
```

Known totals are partial when unknown request counts are nonzero. Reasoning tokens are not added to completion tokens. All failed and interrupted attempts are retained. Individual model distributions and step IDs are in batch.json. Scorer outputs are external evidence, not agent work.
