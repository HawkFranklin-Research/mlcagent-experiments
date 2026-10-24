# Evaluation batch 1

Model, provider, runtime source and rail setting fixed for every study in this batch.
Comparison plan: None
Acceptance outcome and rail effect are reported separately; failed runs are retained.

| Table / replicate | Completed | Setup agrees | Leakage: columns / groups / time | Success | Furthest event / step | Category | Repairs | Requests | Input / output / reasoning | Known cost USD | Wall s | Reported models |
|---|---|---|---|---|---|---|---:|---:|---|---:|---:|---|
| s2_proteome_er / 1 | False | None | None / None / None | False | setup / None | runtime_error:MlcaUserError | 0 | 1 | 234947 / 453 / 103 | None | 77.92983102798462 | {"deepseek-flash": 1} |
| s2_proteome_er / 2 | False | None | None / None / None | False | setup / None | runtime_error:MlcaUserError | 0 | 1 | 234947 / 422 / 120 | None | 77.92898964881897 | {"deepseek-flash": 1} |

## Batch accounting

```json
{
  "requests": 2,
  "known": {
    "input_tokens": 469894,
    "output_tokens": 875,
    "reasoning_tokens": 223,
    "cost_usd": null
  },
  "unknown": {
    "input_tokens": 0,
    "output_tokens": 0,
    "cost_usd": 2
  },
  "partial": true,
  "models": {
    "deepseek-flash": 2
  }
}
```

## Running accounting since batch-01

```json
{
  "requests": 188,
  "known": {
    "input_tokens": 2716700,
    "output_tokens": 482873,
    "reasoning_tokens": 407936,
    "cost_usd": null
  },
  "unknown": {
    "input_tokens": 21,
    "output_tokens": 21,
    "cost_usd": 188
  },
  "partial": true,
  "models": {
    "unavailable": 21,
    "deepseek-flash": 167
  }
}
```

Known totals are partial when unknown request counts are nonzero. Reasoning tokens are not added to completion tokens. All failed and interrupted attempts are retained. Individual model distributions and step IDs are in batch.json. Scorer outputs are external evidence, not agent work.
