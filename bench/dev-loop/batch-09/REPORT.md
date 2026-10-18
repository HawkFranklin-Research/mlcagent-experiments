# Development batch 9

Model, provider, runtime source and rail setting fixed for every study in this batch.
Comparison plan: None
Acceptance outcome and rail effect are reported separately; failed runs are retained.

| Table / replicate | Completed | Setup agrees | Leakage: columns / groups / time | Success | Furthest event / step | Category | Repairs | Requests | Input / output / reasoning | Known cost USD | Wall s | Reported models |
|---|---|---|---|---|---|---|---:|---:|---|---:|---:|---|
| heart_disease_cleveland / 1 | True | True | True / True / True | True | report / 12_report_generate_reports_91WTEWK5 |  | 1 | 38 | 99379 / 89304 / 73307 | None | 481.5988218784332 | {"deepseek-flash": 38} |
| optical_digits / 1 | False | True | True / True / True | False | model / 08_model_train_alternative_NSGSD7TB | runtime_error:ValueError | 4 | 26 | 85469 / 103375 / 93533 | None | 514.0770311355591 | {"deepseek-flash": 26} |
| energy_efficiency / 1 | True | True | True / True / True | True | report / 12_report_generate_reports_Y9NKFDAP |  | 3 | 39 | 105874 / 82741 / 66103 | None | 432.92666363716125 | {"deepseek-flash": 39} |
| bike_sharing_hourly / 1 | True | True | True / True / True | True | report / 12_report_generate_reports_1N3MYBZK |  | 4 | 44 | 123640 / 127017 / 108310 | None | 658.7937881946564 | {"deepseek-flash": 44} |
| anuran_calls / 1 | True | True | True / True / True | True | report / 13_report_generate_reports_GSEJMKTZ |  | 3 | 43 | 116595 / 87063 / 69974 | None | 493.54126381874084 | {"deepseek-flash": 43} |
| online_retail / 1 | False | True | True / True / True | False | model / 07_model_train_baseline_TPF4AZY5 | runtime_error:DTypePromotionError | 3 | 22 | 59453 / 84277 / 74464 | None | 556.5933763980865 | {"deepseek-flash": 22} |
| covertype / 1 | True | True | True / True / True | True | report / 12_report_generate_reports_QTJ9HKW6 |  | 1 | 37 | 96836 / 74897 / 62087 | None | 2534.4972684383392 | {"deepseek-flash": 37} |

## Batch accounting

```json
{
  "requests": 249,
  "known": {
    "input_tokens": 687246,
    "output_tokens": 648674,
    "reasoning_tokens": 547778,
    "cost_usd": null
  },
  "unknown": {
    "input_tokens": 0,
    "output_tokens": 0,
    "cost_usd": 249
  },
  "partial": true,
  "models": {
    "deepseek-flash": 249
  }
}
```

## Running accounting since batch-01

```json
{
  "requests": 249,
  "known": {
    "input_tokens": 687246,
    "output_tokens": 648674,
    "reasoning_tokens": 547778,
    "cost_usd": null
  },
  "unknown": {
    "input_tokens": 0,
    "output_tokens": 0,
    "cost_usd": 249
  },
  "partial": true,
  "models": {
    "deepseek-flash": 249
  }
}
```

Known totals are partial when unknown request counts are nonzero. Reasoning tokens are not added to completion tokens. All failed and interrupted attempts are retained. Individual model distributions and step IDs are in batch.json. Scorer outputs are external evidence, not agent work.
