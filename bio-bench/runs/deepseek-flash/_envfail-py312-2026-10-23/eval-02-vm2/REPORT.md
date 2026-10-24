# Evaluation batch 1

Model, provider, runtime source and rail setting fixed for every study in this batch.
Comparison plan: None
Acceptance outcome and rail effect are reported separately; failed runs are retained.

| Table / replicate | Completed | Setup agrees | Leakage: columns / groups / time | Success | Furthest event / step | Category | Repairs | Requests | Input / output / reasoning | Known cost USD | Wall s | Reported models |
|---|---|---|---|---|---|---|---:|---:|---|---:|---:|---|
| s2_proteome_pr / 1 | False | None | None / None / None | False | setup / None | runtime_error:MlcaUserError | 0 | 1 | 228438 / 404 / 141 | None | 74.53197264671326 | {"deepseek-flash": 1} |
| s2_proteome_pr / 2 | False | None | None / None / None | False | setup / None | runtime_error:MlcaUserError | 0 | 1 | 228438 / 350 / 108 | None | 80.9421615600586 | {"deepseek-flash": 1} |
| s2_proteome_pr / 3 | False | None | None / None / None | False | setup / None | runtime_error:MlcaUserError | 0 | 1 | 228438 / 385 / 107 | None | 76.25215125083923 | {"deepseek-flash": 1} |
| s2_proteome_her2 / 1 | False | None | None / None / None | False | setup / None | runtime_error:MlcaUserError | 0 | 2 | 477113 / 1179 / 398 | None | 86.6845293045044 | {"deepseek-flash": 2} |
| s2_proteome_her2 / 2 | False | None | None / None / None | False | setup / None | runtime_error:MlcaUserError | 0 | 2 | 477064 / 953 / 255 | None | 84.0188193321228 | {"deepseek-flash": 2} |
| s2_proteome_her2 / 3 | False | None | None / None / None | False | setup / None | runtime_error:MlcaUserError | 0 | 1 | 238306 / 833 / 472 | None | 73.73184537887573 | {"deepseek-flash": 1} |

## Batch accounting

```json
{
  "requests": 8,
  "known": {
    "input_tokens": 1877797,
    "output_tokens": 4104,
    "reasoning_tokens": 1481,
    "cost_usd": null
  },
  "unknown": {
    "input_tokens": 0,
    "output_tokens": 0,
    "cost_usd": 8
  },
  "partial": true,
  "models": {
    "deepseek-flash": 8
  }
}
```

## Running accounting since batch-01

```json
{
  "requests": 190,
  "known": {
    "input_tokens": 2482213,
    "output_tokens": 484103,
    "reasoning_tokens": 408488,
    "cost_usd": null
  },
  "unknown": {
    "input_tokens": 21,
    "output_tokens": 21,
    "cost_usd": 190
  },
  "partial": true,
  "models": {
    "unavailable": 21,
    "deepseek-flash": 169
  }
}
```

Known totals are partial when unknown request counts are nonzero. Reasoning tokens are not added to completion tokens. All failed and interrupted attempts are retained. Individual model distributions and step IDs are in batch.json. Scorer outputs are external evidence, not agent work.
