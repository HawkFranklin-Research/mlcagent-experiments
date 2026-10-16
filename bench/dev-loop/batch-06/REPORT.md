# Development batch 6

Development only; model, runtime source and rails fixed for this batch. No comparison with exploratory rounds.

| Table / replicate | Completed | Setup agrees | Leakage: columns / groups / time | Success | Furthest event / step | Category | Repairs | Requests | Input / output / reasoning | Known cost USD | Wall s | Reported models |
|---|---|---|---|---|---|---|---:|---:|---|---:|---:|---|
| optical_digits / 1 | False | True | True / True / True | False | test / 10_test_score_test_WXH3WJR2 | runtime_error:AttributeError | 4 | 34 | 105347 / 80331 / 68676 | 0.087886571 | 479.9653971195221 | {"deepseek/deepseek-v4.1-flash": 34} |
| online_retail / 1 | False | False | None / None / True | False | split / 04_split_freeze_group_JCSCSWBN | runtime_error:MlcaUserError | 0 | 11 | 24782 / 23171 / 19724 | 0.028053997 | 343.88035249710083 | {"deepseek/deepseek-v4.1-flash": 11} |

## Batch accounting

```json
{
  "requests": 45,
  "known": {
    "input_tokens": 130129,
    "output_tokens": 103502,
    "reasoning_tokens": 88400,
    "cost_usd": 0.115940568
  },
  "unknown": {
    "input_tokens": 0,
    "output_tokens": 0,
    "cost_usd": 0
  },
  "partial": false,
  "models": {
    "deepseek/deepseek-v4.1-flash": 45
  }
}
```

## Running accounting since batch-01

```json
{
  "requests": 1127,
  "known": {
    "input_tokens": 2600255,
    "output_tokens": 4447182,
    "reasoning_tokens": 4102628,
    "cost_usd": 4.4820545792
  },
  "unknown": {
    "input_tokens": 0,
    "output_tokens": 0,
    "cost_usd": 0
  },
  "partial": false,
  "models": {
    "deepseek/deepseek-v4.1-flash": 1127
  }
}
```

Known totals are partial when unknown request counts are nonzero. Reasoning tokens are not added to completion tokens. All failed and interrupted attempts are retained. Individual model distributions and step IDs are in batch.json. Scorer outputs are external evidence, not agent work.
