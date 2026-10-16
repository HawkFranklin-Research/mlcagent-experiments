# Development batch 2

Development only; model, runtime source and rails fixed for this batch. No comparison with exploratory rounds.

| Table / replicate | Completed | Setup agrees | Leakage: columns / groups / time | Success | Furthest event / step | Category | Repairs | Requests | Input / output / reasoning | Known cost USD | Wall s | Reported models |
|---|---|---|---|---|---|---|---:|---:|---|---:|---:|---|
| breast_cancer_diagnostic / 1 | False | True | True / True / True | False | validate / 10_validate_audit_validation_E0VZF0JR | runtime_error:ValueError | 5 | 33 | 75214 / 120251 / 108208 | 0.11827847200000001 | 810.3770670890808 | {"deepseek/deepseek-v4.1-flash": 33} |
| heart_disease_cleveland / 1 | True | True | True / True / True | True | report / 12_report_generate_reports_DTAQBMED |  | 2 | 39 | 80255 / 123839 / 112515 | 0.13957652 | 735.6928019523621 | {"deepseek/deepseek-v4.1-flash": 39} |
| adult / 1 | False | True | True / True / True | False | model / 08_model_train_alternative_KSWF1WSF | runtime_error:TypeError | 4 | 26 | 55305 / 118480 / 109600 | 0.11722223600000001 | 803.1900253295898 | {"deepseek/deepseek-v4.1-flash": 26} |
| dry_bean / 1 | False | True | True / True / True | False | test / 11_test_audit_test_BV5NYR9V | runtime_error:ValueError | 3 | 37 | 82040 / 138384 / 126637 | 0.122731512 | 1137.6498131752014 | {"deepseek/deepseek-v4.1-flash": 37} |
| covertype / 1 | False | True | True / True / True | False | validate / 10_validate_audit_validation_SEWF8WK6 | runtime_error:ValueError | 4 | 32 | 72226 / 111535 / 101679 | 0.109756412 | 3362.310930252075 | {"deepseek/deepseek-v4.1-flash": 32} |
| anuran_calls / 1 | False | True | True / True / True | False | validate / 09_validate_score_validation_HYD42CKP | fit_rule | 3 | 28 | 60794 / 128928 / 118795 | 0.134528385 | 721.5734813213348 | {"deepseek/deepseek-v4.1-flash": 28} |
| abalone / 1 | True | True | True / True / True | True | report / 12_report_generate_reports_0SG5A23Q |  | 4 | 40 | 85036 / 169906 / 156490 | 0.168571757 | 996.1879191398621 | {"deepseek/deepseek-v4.1-flash": 40} |
| bike_sharing_hourly / 1 | False | True | True / True / True | False | features / 06_features_retain_features_521S1D5P | runtime_error:KeyError | 3 | 19 | 37562 / 99237 / 93422 | 0.099196942 | 637.5527906417847 | {"deepseek/deepseek-v4.1-flash": 19} |
| energy_efficiency / 1 | True | True | True / True / True | True | report / 12_report_generate_reports_BVWCXES4 |  | 2 | 38 | 79972 / 142782 / 128220 | 0.136400311 | 892.2647457122803 | {"deepseek/deepseek-v4.1-flash": 38} |
| seeds / 1 | False | True | True / True / True | False | model / 07_model_train_baseline_KTDGCWGB | output_truncated | 4 | 23 | 49552 / 98081 / 90318 | 0.108334573 | 559.0333397388458 | {"deepseek/deepseek-v4.1-flash": 23} |
| online_retail / 1 | False | False | None / None / None | False | split / 04_split_freeze_group_5WBNMP7P | runtime_error:MlcaUserError | 3 | 15 | 29574 / 67484 / 62014 | 0.05445411 | 573.3138160705566 | {"deepseek/deepseek-v4.1-flash": 15} |
| optical_digits / 1 | False | True | True / True / True | False | validate / 09_validate_score_validation_KCJJV7DM | output_truncated | 7 | 32 | 86306 / 172091 / 161866 | 0.159888874 | 1003.1807470321655 | {"deepseek/deepseek-v4.1-flash": 32} |

## Batch accounting

```json
{
  "requests": 362,
  "known": {
    "input_tokens": 793836,
    "output_tokens": 1490998,
    "reasoning_tokens": 1369764,
    "cost_usd": 1.468940104
  },
  "unknown": {
    "input_tokens": 0,
    "output_tokens": 0,
    "cost_usd": 0
  },
  "partial": false,
  "models": {
    "deepseek/deepseek-v4.1-flash": 362
  }
}
```

## Running accounting since batch-01

```json
{
  "requests": 535,
  "known": {
    "input_tokens": 1171910,
    "output_tokens": 2205700,
    "reasoning_tokens": 2034216,
    "cost_usd": 2.211723952
  },
  "unknown": {
    "input_tokens": 0,
    "output_tokens": 0,
    "cost_usd": 0
  },
  "partial": false,
  "models": {
    "deepseek/deepseek-v4.1-flash": 535
  }
}
```

Known totals are partial when unknown request counts are nonzero. Reasoning tokens are not added to completion tokens. All failed and interrupted attempts are retained. Individual model distributions and step IDs are in batch.json. Scorer outputs are external evidence, not agent work.
