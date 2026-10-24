# Evaluation batch 1

Model, provider, runtime source and rail setting fixed for every study in this batch.
Comparison plan: None
Acceptance outcome and rail effect are reported separately; failed runs are retained.

| Table / replicate | Completed | Setup agrees | Leakage: columns / groups / time | Success | Furthest event / step | Category | Repairs | Requests | Input / output / reasoning | Known cost USD | Wall s | Reported models |
|---|---|---|---|---|---|---|---:|---:|---|---:|---:|---|
| s2_rna_her2 / 1 | False | None | None / None / None | False | setup / None | runtime_error:MlcaUserError | 0 | 1 | 410490 / 605 / 280 | None | 149.61286330223083 | {"deepseek-flash": 1} |
| s2_rna_her2 / 2 | False | None | None / None / None | False | setup / None | runtime_error:MlcaUserError | 0 | 2 | 821410 / 797 / 165 | None | 149.6260905265808 | {"deepseek-flash": 2} |
| s2_rna_her2 / 3 | False | None | None / None / None | False | setup / None | runtime_error:MlcaUserError | 0 | 1 | 410490 / 597 / 261 | None | 126.93585801124573 | {"deepseek-flash": 1} |

## Batch accounting

```json
{
  "requests": 4,
  "known": {
    "input_tokens": 1642390,
    "output_tokens": 1999,
    "reasoning_tokens": 706,
    "cost_usd": null
  },
  "unknown": {
    "input_tokens": 0,
    "output_tokens": 0,
    "cost_usd": 4
  },
  "partial": true,
  "models": {
    "deepseek-flash": 4
  }
}
```

## Running accounting since batch-01

```json
{
  "requests": 186,
  "known": {
    "input_tokens": 2246806,
    "output_tokens": 481998,
    "reasoning_tokens": 407713,
    "cost_usd": null
  },
  "unknown": {
    "input_tokens": 21,
    "output_tokens": 21,
    "cost_usd": 186
  },
  "partial": true,
  "models": {
    "unavailable": 21,
    "deepseek-flash": 165
  }
}
```

Known totals are partial when unknown request counts are nonzero. Reasoning tokens are not added to completion tokens. All failed and interrupted attempts are retained. Individual model distributions and step IDs are in batch.json. Scorer outputs are external evidence, not agent work.
