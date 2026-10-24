# Evaluation batch 1

Model, provider, runtime source and rail setting fixed for every study in this batch.
Comparison plan: None
Acceptance outcome and rail effect are reported separately; failed runs are retained.

| Table / replicate | Completed | Setup agrees | Leakage: columns / groups / time | Success | Furthest event / step | Category | Repairs | Requests | Input / output / reasoning | Known cost USD | Wall s | Reported models |
|---|---|---|---|---|---|---|---:|---:|---|---:|---:|---|
| s3_pam50 / 1 | False | None | None / None / None | False | setup / None | runtime_error:MlcaUserError | 0 | 1 | 68384 / 410 / 92 | None | 963.8539335727692 | {"deepseek-flash": 1} |
| s3_pam50 / 2 | False | None | None / None / None | False | setup / None | runtime_error:MlcaUserError | 0 | 2 | 137322 / 1809 / 904 | None | 989.9910552501678 | {"deepseek-flash": 2} |
| s3_pam50 / 3 | False | None | None / None / None | False | setup / None | runtime_error:MlcaUserError | 0 | 1 | 68384 / 583 / 285 | None | 881.5861172676086 | {"deepseek-flash": 1} |
| s4_hnscc / 1 | False | None | None / None / None | False | setup / None | runtime_error:MlcaUserError | 0 | 1 | 8929 / 524 / 227 | None | 25.4496488571167 | {"deepseek-flash": 1} |
| s4_hnscc / 2 | False | None | None / None / None | False | setup / None | runtime_error:MlcaUserError | 0 | 1 | 8929 / 594 / 323 | None | 21.610594987869263 | {"deepseek-flash": 1} |
| s4_hnscc / 3 | False | None | None / None / None | False | setup / None | runtime_error:MlcaUserError | 0 | 2 | 18300 / 876 / 188 | None | 26.21018886566162 | {"deepseek-flash": 2} |

## Batch accounting

```json
{
  "requests": 8,
  "known": {
    "input_tokens": 310248,
    "output_tokens": 4796,
    "reasoning_tokens": 2019,
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
    "input_tokens": 914664,
    "output_tokens": 484795,
    "reasoning_tokens": 409026,
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
