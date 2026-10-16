# Development batch 4

Development only; model, runtime source and rails fixed for this batch. No comparison with exploratory rounds.

| Table / replicate | Completed | Setup agrees | Leakage: columns / groups / time | Success | Furthest event / step | Category | Repairs | Requests | Input / output / reasoning | Known cost USD | Wall s | Reported models |
|---|---|---|---|---|---|---|---:|---:|---|---:|---:|---|
| anuran_calls / 1 | True | True | True / True / True | True | report / 12_report_generate_reports_B8TFW1VW |  | 5 | 45 | 108561 / 188666 / 175355 | 0.19894710300000001 | 1479.4524638652802 | {"deepseek/deepseek-v4.1-flash": 45} |
| bike_sharing_hourly / 1 | True | False | True / True / False | False | report / 12_report_generate_reports_XMQRBXHH | setup_disagreement | 2 | 40 | 88033 / 148842 / 137547 | 0.151602591 | 848.8898866176605 | {"deepseek/deepseek-v4.1-flash": 40} |
| seeds / 1 | True | True | True / True / True | True | report / 12_report_generate_reports_3MJYHTVF |  | 6 | 44 | 104407 / 180471 / 165212 | 0.174826728 | 1090.5697484016418 | {"deepseek/deepseek-v4.1-flash": 44} |
| optical_digits / 1 | False | True | None / True / True | False | eda / 03_eda_describe_distributions_17EHR8F4 | output_truncated | 2 | 11 | 33959 / 63667 / 63195 | 0.043988428 | 352.12674832344055 | {"deepseek/deepseek-v4.1-flash": 11} |
| online_retail / 1 | True | False | True / True / True | False | report / 12_report_generate_reports_CCXPDYKF | setup_disagreement | 6 | 44 | 98549 / 188299 / 175649 | 0.17481028 | 1554.55859375 | {"deepseek/deepseek-v4.1-flash": 44} |

## Batch accounting

```json
{
  "requests": 184,
  "known": {
    "input_tokens": 433509,
    "output_tokens": 769945,
    "reasoning_tokens": 716958,
    "cost_usd": 0.74417513
  },
  "unknown": {
    "input_tokens": 0,
    "output_tokens": 0,
    "cost_usd": 0
  },
  "partial": false,
  "models": {
    "deepseek/deepseek-v4.1-flash": 184
  }
}
```

## Running accounting since batch-01

```json
{
  "requests": 998,
  "known": {
    "input_tokens": 2269317,
    "output_tokens": 4127847,
    "reasoning_tokens": 3827577,
    "cost_usd": 4.1218700654000004
  },
  "unknown": {
    "input_tokens": 0,
    "output_tokens": 0,
    "cost_usd": 0
  },
  "partial": false,
  "models": {
    "deepseek/deepseek-v4.1-flash": 998
  }
}
```

Known totals are partial when unknown request counts are nonzero. Reasoning tokens are not added to completion tokens. All failed and interrupted attempts are retained. Individual model distributions and step IDs are in batch.json. Scorer outputs are external evidence, not agent work.
