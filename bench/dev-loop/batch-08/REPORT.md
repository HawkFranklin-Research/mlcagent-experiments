# Development batch 8

Development only; model, runtime source and rails fixed for this batch. No comparison with exploratory rounds.

| Table / replicate | Completed | Setup agrees | Leakage: columns / groups / time | Success | Furthest event / step | Category | Repairs | Requests | Input / output / reasoning | Known cost USD | Wall s | Reported models |
|---|---|---|---|---|---|---|---:|---:|---|---:|---:|---|
| online_retail / 1 | True | True | True / True / True | True | report / 12_report_generate_reports_1D6XAZA1 |  | 10 | 51 | 151029 / 170486 / 149638 | 0.09665747820000001 | 1106.8145835399628 | {"deepseek/deepseek-v4.1-flash": 51} |

## Batch accounting

```json
{
  "requests": 51,
  "known": {
    "input_tokens": 151029,
    "output_tokens": 170486,
    "reasoning_tokens": 149638,
    "cost_usd": 0.09665747820000001
  },
  "unknown": {
    "input_tokens": 0,
    "output_tokens": 0,
    "cost_usd": 0
  },
  "partial": false,
  "models": {
    "deepseek/deepseek-v4.1-flash": 51
  }
}
```

## Running accounting since batch-01

```json
{
  "requests": 1265,
  "known": {
    "input_tokens": 3001991,
    "output_tokens": 4820251,
    "reasoning_tokens": 4425729,
    "cost_usd": 4.798535326024
  },
  "unknown": {
    "input_tokens": 0,
    "output_tokens": 0,
    "cost_usd": 0
  },
  "partial": false,
  "models": {
    "deepseek/deepseek-v4.1-flash": 1265
  }
}
```

Known totals are partial when unknown request counts are nonzero. Reasoning tokens are not added to completion tokens. All failed and interrupted attempts are retained. Individual model distributions and step IDs are in batch.json. Scorer outputs are external evidence, not agent work.
