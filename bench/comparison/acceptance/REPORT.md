# Acceptance batch acceptance

Model, provider, runtime source and rail setting fixed for every study in this batch.
Comparison plan: c9d5bc8715b8dcd02e1a9611dcd1036f168de56bb779b0ade8b95d83a994dc2f
Acceptance outcome and rail effect are reported separately; failed runs are retained.

| Table / replicate | Completed | Setup agrees | Leakage: columns / groups / time | Success | Furthest event / step | Category | Repairs | Requests | Input / output / reasoning | Known cost USD | Wall s | Reported models |
|---|---|---|---|---|---|---|---:|---:|---|---:|---:|---|
| breast_cancer_diagnostic / 1 | True | True | True / True / True | True | report / 12_report_generate_reports_6PYSPBRP |  | 3 | 42 | 119025 / 113045 / 94264 | None | 656.4589374065399 | {"deepseek-flash": 42} |
| breast_cancer_diagnostic / 2 | True | True | True / True / True | True | report / 12_report_generate_reports_9E39D9YJ |  | 1 | 38 | 98429 / 74915 / 61335 | None | 456.7062828540802 | {"deepseek-flash": 38} |
| breast_cancer_diagnostic / 3 | True | True | True / True / True | True | report / 12_report_generate_reports_E9MY39X7 |  | 0 | 36 | 90053 / 71796 / 57696 | None | 456.7307288646698 | {"deepseek-flash": 36} |
| heart_disease_cleveland / 1 | False | None | None / None / None | False | setup / None | invalid_inference | 0 | 2 | 10436 / 1145 / 181 | None | 44.76677346229553 | {"deepseek-flash": 2} |
| heart_disease_cleveland / 2 | True | True | True / True / True | True | report / 12_report_generate_reports_1R2DCMSR |  | 1 | 39 | 104308 / 87148 / 71205 | None | 503.2640175819397 | {"deepseek-flash": 39} |
| heart_disease_cleveland / 3 | True | True | True / True / True | True | report / 12_report_generate_reports_2TV6W3J3 |  | 2 | 40 | 106184 / 105684 / 90903 | None | 593.1654717922211 | {"deepseek-flash": 40} |
| adult / 1 | True | True | True / True / True | True | report / 12_report_generate_reports_0ES5AN8N |  | 3 | 40 | 107468 / 118624 / 101865 | None | 759.027585029602 | {"deepseek-flash": 40} |
| adult / 2 | True | True | True / True / True | True | report / 12_report_generate_reports_XZ4GM22E |  | 2 | 38 | 96586 / 92707 / 78517 | None | 619.4268865585327 | {"deepseek-flash": 38} |
| adult / 3 | True | True | True / True / True | True | report / 12_report_generate_reports_9FKNR4X1 |  | 2 | 41 | 109734 / 103655 / 88381 | None | 694.2748534679413 | {"deepseek-flash": 41} |
| dry_bean / 1 | True | True | True / True / True | True | report / 12_report_generate_reports_9R7KQD49 |  | 2 | 40 | 104838 / 57441 / 44905 | None | 424.2113604545593 | {"deepseek-flash": 40} |
| dry_bean / 2 | True | True | True / True / True | True | report / 12_report_generate_reports_ZXA6JYGX |  | 0 | 39 | 103420 / 74574 / 61085 | None | 598.7085618972778 | {"deepseek-flash": 39} |
| dry_bean / 3 | True | True | True / True / True | True | report / 12_report_generate_reports_WCDNYZFK |  | 2 | 40 | 107611 / 79314 / 64822 | None | 554.3964471817017 | {"deepseek-flash": 40} |
| covertype / 1 | False | None | None / None / None | False | setup / None | worker_exception | 0 | 0 | 0 / 0 / 0 | 0 | 2902.8616461753845 | {} |
| covertype / 2 | False | None | None / None / None | False | setup / None | runtime_error:OperationalError | 0 | 1 | 7524 / 379 / 88 | None | 1096.7364892959595 | {"deepseek-flash": 1} |
| covertype / 3 | False | True | True / True / True | False | model / 08_model_train_alternative_SQ9B8BN3 | study_deadline | 0 | 22 | 54161 / 55364 / 47250 | None | 48906.54311919212 | {"deepseek-flash": 22} |
| anuran_calls / 1 | False | False | True / True / True | False | validate / 10_validate_score_validation_WQ0EE977 | runtime_error:ValueError | 4 | 34 | 91694 / 83855 / 70355 | None | 529.4158692359924 | {"deepseek-flash": 34} |
| anuran_calls / 2 | True | False | True / True / True | False | report / 13_report_generate_reports_2QRXW47B | setup_disagreement | 3 | 42 | 111721 / 118729 / 100054 | None | 821.386109828949 | {"deepseek-flash": 42} |
| anuran_calls / 3 | True | True | False / True / True | False | report / 13_report_generate_reports_SHTVNG2T | leakage_failed | 2 | 42 | 108627 / 85642 / 70431 | None | 539.3938291072845 | {"deepseek-flash": 42} |
| abalone / 1 | True | True | True / True / True | True | report / 12_report_generate_reports_AH77TETM |  | 2 | 38 | 97033 / 99165 / 83349 | None | 609.1218709945679 | {"deepseek-flash": 38} |
| abalone / 2 | True | True | True / True / True | True | report / 12_report_generate_reports_P8DRNZCJ |  | 3 | 39 | 102143 / 101323 / 83252 | None | 562.321281671524 | {"deepseek-flash": 39} |
| abalone / 3 | True | True | True / True / True | True | report / 12_report_generate_reports_JXW763N0 |  | 1 | 37 | 92350 / 82207 / 67918 | None | 513.7800657749176 | {"deepseek-flash": 37} |
| bike_sharing_hourly / 1 | False | False | None / True / None | False | split / 04_split_freeze_time_M646CV6R | runtime_error:MlcaUserError | 0 | 10 | 23966 / 13779 / 10885 | None | 105.56973528862 | {"deepseek-flash": 10} |
| bike_sharing_hourly / 2 | False | False | None / True / None | False | split / 04_split_freeze_time_D3RRC3JW | runtime_error:MlcaUserError | 0 | 11 | 27379 / 14479 / 10973 | None | 115.44114899635315 | {"deepseek-flash": 11} |
| bike_sharing_hourly / 3 | True | True | True / True / True | True | report / 12_report_generate_reports_JPP2Z1Q2 |  | 3 | 39 | 107025 / 122110 / 105381 | None | 1416.0506129264832 | {"deepseek-flash": 39} |
| energy_efficiency / 1 | True | True | True / True / True | True | report / 12_report_generate_reports_7NJ98R26 |  | 2 | 42 | 114690 / 82218 / 66634 | None | 1377.8727197647095 | {"deepseek-flash": 42} |
| energy_efficiency / 2 | True | True | True / True / True | True | report / 12_report_generate_reports_45TNMHSB |  | 1 | 38 | 96228 / 88606 / 73996 | None | 1652.5893490314484 | {"deepseek-flash": 38} |
| energy_efficiency / 3 | False | True | False / True / True | False | test / 10_test_score_test_4SSVV8KS | runtime_error:ValueError | 4 | 35 | 103203 / 125121 / 108346 | None | 1784.6353125572205 | {"deepseek-flash": 35} |
| seeds / 1 | True | True | True / True / True | True | report / 12_report_generate_reports_54EJT9RC |  | 1 | 39 | 101958 / 81372 / 65269 | None | 1545.9433023929596 | {"deepseek-flash": 39} |
| seeds / 2 | True | True | True / True / True | True | report / 12_report_generate_reports_CDN8V6CA |  | 1 | 39 | 101278 / 92607 / 78972 | None | 2041.6339571475983 | {"deepseek-flash": 39} |
| seeds / 3 | True | True | True / True / True | True | report / 12_report_generate_reports_35DRJDVG |  | 0 | 40 | 105079 / 78861 / 65096 | None | 1993.410948753357 | {"deepseek-flash": 40} |
| online_retail / 1 | False | True | True / True / True | False | model / 08_model_train_alternative_4PEDXQ5Q | runtime_error:ValueError | 3 | 27 | 71819 / 92076 / 82141 | None | 632.5324809551239 | {"deepseek-flash": 27} |
| online_retail / 2 | False | True | None / None / True | False | split / 04_split_freeze_stratified_8S587J5T | runtime_error:MlcaUserError | 0 | 13 | 35242 / 18670 / 14216 | None | 252.34039568901062 | {"deepseek-flash": 13} |
| online_retail / 3 | False | True | None / None / True | False | split / 04_split_freeze_group_P99NV8B7 | runtime_error:MlcaUserError | 0 | 11 | 27904 / 11064 / 7636 | None | 206.09081864356995 | {"deepseek-flash": 11} |
| optical_digits / 1 | False | None | None / None / None | False | setup / None | runtime_error:MlcaUserError | 0 | 2 | 34571 / 1426 / 558 | None | 1043.7155458927155 | {"deepseek-flash": 2} |
| optical_digits / 2 | True | True | True / True / True | True | report / 12_report_generate_reports_YQ7GJY1T |  | 2 | 42 | 127002 / 102901 / 85848 | None | 593.4525322914124 | {"deepseek-flash": 42} |
| optical_digits / 3 | True | True | True / True / True | True | report / 12_report_generate_reports_55XFSVXE |  | 0 | 38 | 122482 / 75361 / 61975 | None | 424.2604696750641 | {"deepseek-flash": 38} |

## Batch accounting

```json
{
  "requests": 1116,
  "known": {
    "input_tokens": 3023171,
    "output_tokens": 2607363,
    "reasoning_tokens": 2175782,
    "cost_usd": null
  },
  "unknown": {
    "input_tokens": 0,
    "output_tokens": 0,
    "cost_usd": 1116
  },
  "partial": true,
  "models": {
    "deepseek-flash": 1116
  }
}
```

## Running accounting since batch-01

```json
{
  "requests": 1116,
  "known": {
    "input_tokens": 3023171,
    "output_tokens": 2607363,
    "reasoning_tokens": 2175782,
    "cost_usd": null
  },
  "unknown": {
    "input_tokens": 0,
    "output_tokens": 0,
    "cost_usd": 1116
  },
  "partial": true,
  "models": {
    "deepseek-flash": 1116
  }
}
```

Known totals are partial when unknown request counts are nonzero. Reasoning tokens are not added to completion tokens. All failed and interrupted attempts are retained. Individual model distributions and step IDs are in batch.json. Scorer outputs are external evidence, not agent work.
