# Development batch 1

Development only; model, runtime source and rails fixed for this batch. No comparison with exploratory rounds.

| Table / replicate | Completed | Setup agrees | Leakage: columns / groups / time | Success | Furthest event / step | Category | Repairs | Requests | Input / output / reasoning | Known cost USD | Wall s | Reported models |
|---|---|---|---|---|---|---|---:|---:|---|---:|---:|---|
| breast_cancer_diagnostic / 1 | False | True | True / True / True | False | features / 06_features_select_features_THDS477D | output_truncated | 2 | 19 | 38201 / 73825 / 67073 | 0.068450612 | 509.7602984905243 | {"deepseek/deepseek-v4.1-flash": 19} |
| heart_disease_cleveland / 1 | False | True | None / True / True | False | preprocess / 05_preprocess_fit_preprocessing_VHAA70AD | runtime_error:ValueError | 3 | 17 | 35067 / 90756 / 86715 | 0.102003558 | 410.8138098716736 | {"deepseek/deepseek-v4.1-flash": 17} |
| adult / 1 | False | None | None / None / None | False | setup / None | invalid_inference | 0 | 2 | 8153 / 4464 / 4243 | 0.004983033 | 31.00366735458374 | {"deepseek/deepseek-v4.1-flash": 2} |
| dry_bean / 1 | False | None | None / None / None | False | setup / None | invalid_inference | 0 | 2 | 7883 / 2911 / 2732 | 0.003216231 | 24.760555505752563 | {"deepseek/deepseek-v4.1-flash": 2} |
| covertype / 1 | False | True | None / True / True | False | data / 02_data_audit_schema_M9N02NGG | interrupted | 0 | 6 | 12978 / 12291 / 11159 | 0.010999033 | 1517.6563975811005 | {"deepseek/deepseek-v4.1-flash": 6} |
| anuran_calls / 1 | False | False | False / True / True | False | model / 07_model_train_baseline_6BBET7SE | runtime_error:AttributeError | 6 | 26 | 60985 / 134987 / 125160 | 0.137159374 | 791.156566619873 | {"deepseek/deepseek-v4.1-flash": 26} |
| abalone / 1 | False | None | None / None / None | False | setup / None | invalid_inference | 0 | 2 | 5850 / 4512 / 4359 | 0.004423962 | 28.11287546157837 | {"deepseek/deepseek-v4.1-flash": 2} |
| bike_sharing_hourly / 1 | False | False | True / True / False | False | features / 06_features_retain_features_2SJ6FCFF | output_truncated | 3 | 20 | 39851 / 95586 / 89469 | 0.093271447 | 492.0913915634155 | {"deepseek/deepseek-v4.1-flash": 20} |
| energy_efficiency / 1 | False | True | True / True / True | False | model / 07_model_train_baseline_Z1409K6D | runtime_error:AttributeError | 3 | 24 | 48250 / 95575 / 88890 | 0.109335792 | 533.7191264629364 | {"deepseek/deepseek-v4.1-flash": 24} |
| seeds / 1 | False | True | True / True / True | False | model / 07_model_train_baseline_XHYG11BX | output_truncated | 3 | 23 | 45075 / 94911 / 89396 | 0.101570362 | 440.96786236763 | {"deepseek/deepseek-v4.1-flash": 23} |
| online_retail / 1 | False | False | None / None / None | False | eda / 03_eda_build_entities_VWBZMCWK | runtime_error:ValueError | 2 | 11 | 21630 / 30786 / 27277 | 0.030824342 | 366.9233660697937 | {"deepseek/deepseek-v4.1-flash": 11} |
| optical_digits / 1 | False | True | True / True / True | False | model / 07_model_train_baseline_ZXWXR3BX | runtime_error:ValueError | 2 | 21 | 54151 / 74098 / 67979 | 0.076546102 | 439.53487253189087 | {"deepseek/deepseek-v4.1-flash": 21} |

## Batch accounting

```json
{
  "requests": 173,
  "known": {
    "input_tokens": 378074,
    "output_tokens": 714702,
    "reasoning_tokens": 664452,
    "cost_usd": 0.742783848
  },
  "unknown": {
    "input_tokens": 0,
    "output_tokens": 0,
    "cost_usd": 0
  },
  "partial": false,
  "models": {
    "deepseek/deepseek-v4.1-flash": 173
  }
}
```

## Running accounting since batch-01

```json
{
  "requests": 173,
  "known": {
    "input_tokens": 378074,
    "output_tokens": 714702,
    "reasoning_tokens": 664452,
    "cost_usd": 0.742783848
  },
  "unknown": {
    "input_tokens": 0,
    "output_tokens": 0,
    "cost_usd": 0
  },
  "partial": false,
  "models": {
    "deepseek/deepseek-v4.1-flash": 173
  }
}
```

Known totals are partial when unknown request counts are nonzero. Reasoning tokens are not added to completion tokens. All failed and interrupted attempts are retained. Individual model distributions and step IDs are in batch.json. Scorer outputs are external evidence, not agent work.

## Review annotations

See [REVIEW.md](REVIEW.md). Covertype was an early implementing-agent infrastructure intervention, not a model failure; the original recorded category remains `interrupted`. Setup agreement and leakage checks are separate from workflow completion.
