# Development batch 3

Development only; model, runtime source and rails fixed for this batch. No comparison with exploratory rounds.

| Table / replicate | Completed | Setup agrees | Leakage: columns / groups / time | Success | Furthest event / step | Category | Repairs | Requests | Input / output / reasoning | Known cost USD | Wall s | Reported models |
|---|---|---|---|---|---|---|---:|---:|---|---:|---:|---|
| breast_cancer_diagnostic / 1 | True | True | True / True / True | True | report / 12_report_generate_reports_203JPVJE |  | 2 | 38 | 85463 / 126022 / 114422 | 0.119702411 | 877.6848051548004 | {"deepseek/deepseek-v4.1-flash": 38} |
| adult / 1 | True | True | True / True / True | True | report / 12_report_generate_reports_YJB96YBW |  | 3 | 39 | 87571 / 155750 / 147492 | 0.162576414 | 1150.30184674263 | {"deepseek/deepseek-v4.1-flash": 39} |
| dry_bean / 1 | True | True | True / True / True | True | report / 12_report_generate_reports_FPSAZDTE |  | 2 | 40 | 93440 / 130322 / 118918 | 0.127990118 | 900.5829374790192 | {"deepseek/deepseek-v4.1-flash": 40} |
| covertype / 1 | True | True | True / True / True | True | report / 12_report_generate_reports_TEPYVHGM |  | 4 | 40 | 96292 / 135784 / 123874 | 0.137864959 | 4236.201169013977 | {"deepseek/deepseek-v4.1-flash": 40} |
| anuran_calls / 1 | False | True | True / True / True | False | validate / 09_validate_score_validation_3MRHWYYC | output_truncated | 4 | 29 | 68785 / 140432 / 133028 | 0.144741922 | 727.7379093170166 | {"deepseek/deepseek-v4.1-flash": 29} |
| bike_sharing_hourly / 1 | False | False | True / True / False | False | preprocess / 11_preprocess_fit_preprocessing_A3X0C743 | output_truncated | 7 | 38 | 90394 / 185657 / 174880 | 0.211677393 | 954.0747034549713 | {"deepseek/deepseek-v4.1-flash": 38} |
| seeds / 1 | False | True | True / True / True | False | model / 07_model_train_baseline_C14KNZHQ | sandbox_denied | 4 | 23 | 52612 / 105104 / 98915 | 0.116933138 | 624.7851555347443 | {"deepseek/deepseek-v4.1-flash": 23} |
| optical_digits / 1 | False | True | True / True / True | False | validate / 09_validate_score_validation_BT6AKQ6V | runtime_error:AttributeError | 7 | 32 | 89341 / 173131 / 164874 | 0.1444846284 | 995.107488155365 | {"deepseek/deepseek-v4.1-flash": 32} |

## Batch accounting

```json
{
  "requests": 279,
  "known": {
    "input_tokens": 663898,
    "output_tokens": 1152202,
    "reasoning_tokens": 1076403,
    "cost_usd": 1.1659709834
  },
  "unknown": {
    "input_tokens": 0,
    "output_tokens": 0,
    "cost_usd": 0
  },
  "partial": false,
  "models": {
    "deepseek/deepseek-v4.1-flash": 279
  }
}
```

## Running accounting since batch-01

```json
{
  "requests": 814,
  "known": {
    "input_tokens": 1835808,
    "output_tokens": 3357902,
    "reasoning_tokens": 3110619,
    "cost_usd": 3.3776949354
  },
  "unknown": {
    "input_tokens": 0,
    "output_tokens": 0,
    "cost_usd": 0
  },
  "partial": false,
  "models": {
    "deepseek/deepseek-v4.1-flash": 814
  }
}
```

Known totals are partial when unknown request counts are nonzero. Reasoning tokens are not added to completion tokens. All failed and interrupted attempts are retained. Individual model distributions and step IDs are in batch.json. Scorer outputs are external evidence, not agent work.
