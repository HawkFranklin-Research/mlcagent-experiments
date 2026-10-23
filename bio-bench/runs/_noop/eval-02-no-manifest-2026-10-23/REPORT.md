# Evaluation batch 1

Model, provider, runtime source and rail setting fixed for every study in this batch.
Comparison plan: None
Acceptance outcome and rail effect are reported separately; failed runs are retained.

| Table / replicate | Completed | Setup agrees | Leakage: columns / groups / time | Success | Furthest event / step | Category | Repairs | Requests | Input / output / reasoning | Known cost USD | Wall s | Reported models |
|---|---|---|---|---|---|---|---:|---:|---|---:|---:|---|

## Batch accounting

```json
{
  "requests": 0,
  "known": {
    "input_tokens": 0,
    "output_tokens": 0,
    "reasoning_tokens": 0,
    "cost_usd": 0
  },
  "unknown": {
    "input_tokens": 0,
    "output_tokens": 0,
    "cost_usd": 0
  },
  "partial": false,
  "models": {}
}
```

## Running accounting since batch-01

```json
{
  "requests": 182,
  "known": {
    "input_tokens": 604416,
    "output_tokens": 479999,
    "reasoning_tokens": 407007,
    "cost_usd": null
  },
  "unknown": {
    "input_tokens": 21,
    "output_tokens": 21,
    "cost_usd": 182
  },
  "partial": true,
  "models": {
    "unavailable": 21,
    "deepseek-flash": 161
  }
}
```

Known totals are partial when unknown request counts are nonzero. Reasoning tokens are not added to completion tokens. All failed and interrupted attempts are retained. Individual model distributions and step IDs are in batch.json. Scorer outputs are external evidence, not agent work.
