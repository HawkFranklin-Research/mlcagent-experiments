# Acceptance batch acceptance

Model, provider, runtime source and rail setting fixed for every study in this batch.
Comparison plan: 630a2c978a05b22aed987178f2be5fc5de689b40a419e07396671cf91562b1b8
Acceptance outcome and rail effect are reported separately; failed runs are retained.

| Table / replicate | Completed | Setup agrees | Leakage: columns / groups / time | Success | Furthest event / step | Category | Repairs | Requests | Input / output / reasoning | Known cost USD | Wall s | Reported models |
|---|---|---|---|---|---|---|---:|---:|---|---:|---:|---|
| breast_cancer_diagnostic / 1 | True | True | True / True / True | True | report / 12_report_generate_reports_PTX1GB58 |  | 2 | 39 | 107304 / 109484 / 92709 | None | 582.9526228904724 | {"deepseek-flash": 39} |
| breast_cancer_diagnostic / 2 | True | True | True / True / True | True | report / 13_report_generate_reports_5YNEVYRX |  | 4 | 43 | 115923 / 99635 / 82871 | None | 531.565714597702 | {"deepseek-flash": 43} |
| breast_cancer_diagnostic / 3 | True | True | True / True / True | True | report / 12_report_generate_reports_BJQKSF9H |  | 0 | 36 | 91843 / 77793 / 64127 | None | 413.98914313316345 | {"deepseek-flash": 36} |
| heart_disease_cleveland / 1 | True | True | True / True / True | True | report / 12_report_generate_reports_GRCZ33D8 |  | 0 | 36 | 92154 / 78470 / 65350 | None | 411.70565152168274 | {"deepseek-flash": 36} |
| heart_disease_cleveland / 2 | True | False | True / True / True | False | report / 12_report_generate_reports_Q77W0XY1 | setup_disagreement | 1 | 41 | 114347 / 92199 / 75559 | None | 475.6700723171234 | {"deepseek-flash": 41} |
| heart_disease_cleveland / 3 | True | False | True / True / True | False | report / 12_report_generate_reports_VBW0K98G | setup_disagreement | 0 | 36 | 92530 / 85741 / 72146 | None | 434.24064087867737 | {"deepseek-flash": 36} |
| adult / 1 | True | True | True / True / True | True | report / 12_report_generate_reports_E7FN7865 |  | 2 | 38 | 101716 / 85675 / 70834 | None | 544.7563779354095 | {"deepseek-flash": 38} |
| adult / 2 | True | True | True / True / True | True | report / 12_report_generate_reports_F98GGE59 |  | 5 | 45 | 135410 / 141988 / 119994 | None | 816.6918721199036 | {"deepseek-flash": 45} |
| adult / 3 | False | True | None / True / True | False | preprocess / 05_preprocess_fit_preprocessing_S660K829 | output_truncated | 2 | 15 | 39268 / 62029 / 56700 | None | 329.01346039772034 | {"deepseek-flash": 15} |
| dry_bean / 1 | False | True | True / True / True | False | interpret / 11_interpret_explain_model_J11J9MK4 | runtime_error:TypeError | 2 | 35 | 96293 / 102447 / 88645 | None | 551.6953418254852 | {"deepseek-flash": 35} |
| dry_bean / 2 | True | True | True / True / True | True | report / 12_report_generate_reports_780BNJSP |  | 0 | 36 | 90355 / 81447 / 67554 | None | 515.1926321983337 | {"deepseek-flash": 36} |
| dry_bean / 3 | True | True | True / True / True | True | report / 12_report_generate_reports_Q51WRGPQ |  | 0 | 37 | 92996 / 72139 / 58277 | None | 496.3426356315613 | {"deepseek-flash": 37} |
| covertype / 1 | False | True | True / True / True | False | interpret / 11_interpret_explain_model_60902DF4 | timeout | 4 | 37 | 106787 / 111517 / 94621 | None | 3663.37610912323 | {"deepseek-flash": 37} |
| covertype / 2 | False | True | None / True / True | False | preprocess / 05_preprocess_fit_preprocessing_CHC2E5Z0 | output_truncated | 2 | 15 | 41620 / 55445 / 50795 | None | 2242.6302468776703 | {"deepseek-flash": 15} |
| covertype / 3 | False | None | None / None / None | False | setup / None | runtime_error:OperationalError | 0 | 1 | 7581 / 326 / 40 | None | 459.253769159317 | {"deepseek-flash": 1} |
| anuran_calls / 1 | True | True | True / True / True | True | report / 12_report_generate_reports_TZ1XKFRT |  | 2 | 40 | 109525 / 108123 / 91880 | None | 575.2215473651886 | {"deepseek-flash": 40} |
| anuran_calls / 2 | True | True | True / True / True | True | report / 12_report_generate_reports_ZPMXRBF6 |  | 0 | 38 | 100710 / 95080 / 79526 | None | 548.8401279449463 | {"deepseek-flash": 38} |
| anuran_calls / 3 | True | True | True / True / True | True | report / 12_report_generate_reports_2MRSJ4XJ |  | 3 | 40 | 110268 / 104036 / 86198 | None | 554.4081423282623 | {"deepseek-flash": 40} |
| abalone / 1 | True | True | True / True / True | True | report / 12_report_generate_reports_M5HS6H08 |  | 1 | 37 | 94951 / 79658 / 64433 | None | 441.1057872772217 | {"deepseek-flash": 37} |
| abalone / 2 | True | True | True / True / True | True | report / 12_report_generate_reports_R9QKAD55 |  | 3 | 39 | 105286 / 118060 / 99991 | None | 588.2850301265717 | {"deepseek-flash": 39} |
| abalone / 3 | True | True | True / True / True | True | report / 12_report_generate_reports_JY37QNCW |  | 3 | 44 | 125109 / 123486 / 107677 | None | 637.5248503684998 | {"deepseek-flash": 44} |
| bike_sharing_hourly / 1 | False | True | True / True / True | False | features / 06_features_retain_features_35P4XP7P | output_truncated | 2 | 18 | 46564 / 55501 / 48998 | None | 322.39539551734924 | {"deepseek-flash": 18} |
| bike_sharing_hourly / 2 | False | True | True / True / True | False | features / 06_features_retain_features_18G48GNV | runtime_error:ValueError | 3 | 19 | 50552 / 60210 / 54597 | None | 323.8747444152832 | {"deepseek-flash": 19} |
| bike_sharing_hourly / 3 | False | True | True / True / True | False | model / 07_model_train_baseline_FQWWDMS5 | runtime_error:ValueError | 5 | 24 | 68137 / 91506 / 82041 | None | 492.07222509384155 | {"deepseek-flash": 24} |
| energy_efficiency / 1 | True | True | True / True / True | True | report / 12_report_generate_reports_T0RPHM29 |  | 0 | 37 | 93389 / 71577 / 58111 | None | 394.94929027557373 | {"deepseek-flash": 37} |
| energy_efficiency / 2 | True | True | True / True / True | True | report / 12_report_generate_reports_WFYB3QK9 |  | 2 | 39 | 102184 / 113749 / 99117 | None | 584.2028369903564 | {"deepseek-flash": 39} |
| energy_efficiency / 3 | True | True | True / True / True | True | report / 12_report_generate_reports_SQFMCAMQ |  | 0 | 36 | 89716 / 66886 / 54364 | None | 396.8774838447571 | {"deepseek-flash": 36} |
| seeds / 1 | False | None | None / None / None | False | setup / None | external_scoring_error | 0 | 5 | 12936 / 2841 / 1778 | None | None | {"deepseek-flash": 5} |
| seeds / 2 | True | True | True / True / True | True | report / 12_report_generate_reports_8B3VYPCS |  | 0 | 37 | 94673 / 85179 / 70640 | None | 444.31367588043213 | {"deepseek-flash": 37} |
| seeds / 3 | True | True | True / True / True | True | report / 12_report_generate_reports_0REM8V0R |  | 2 | 38 | 99861 / 93195 / 78051 | None | 482.72527623176575 | {"deepseek-flash": 38} |
| online_retail / 1 | True | True | True / True / True | True | report / 12_report_generate_reports_ZF5Z92SJ |  | 4 | 40 | 105917 / 120132 / 104747 | None | 797.4348165988922 | {"deepseek-flash": 40} |
| online_retail / 2 | True | True | True / True / True | True | report / 12_report_generate_reports_01Q5FGZQ |  | 1 | 38 | 96058 / 100205 / 85272 | None | 688.6132171154022 | {"deepseek-flash": 38} |
| online_retail / 3 | False | None | None / None / None | False | setup / None | runtime_error:OperationalError | 0 | 1 | 4344 / 1133 / 704 | None | 100.35208535194397 | {"deepseek-flash": 1} |
| optical_digits / 1 | True | True | True / True / True | True | report / 12_report_generate_reports_3F7G7537 |  | 4 | 41 | 131190 / 139937 / 119132 | None | 766.5392990112305 | {"deepseek-flash": 41} |
| optical_digits / 2 | True | True | True / True / True | True | report / 12_report_generate_reports_R4QBJ2ED |  | 4 | 41 | 127361 / 106504 / 92169 | None | 577.2416632175446 | {"deepseek-flash": 41} |
| optical_digits / 3 | True | True | True / True / True | True | report / 12_report_generate_reports_T6MRV75E |  | 2 | 40 | 123046 / 115598 / 98686 | None | 644.03906083107 | {"deepseek-flash": 40} |

## Batch accounting

```json
{
  "requests": 1182,
  "known": {
    "input_tokens": 3217904,
    "output_tokens": 3108931,
    "reasoning_tokens": 2638334,
    "cost_usd": null
  },
  "unknown": {
    "input_tokens": 0,
    "output_tokens": 0,
    "cost_usd": 1182
  },
  "partial": true,
  "models": {
    "deepseek-flash": 1182
  }
}
```

## Running accounting since batch-01

```json
{
  "requests": 2298,
  "known": {
    "input_tokens": 6241075,
    "output_tokens": 5716294,
    "reasoning_tokens": 4814116,
    "cost_usd": null
  },
  "unknown": {
    "input_tokens": 0,
    "output_tokens": 0,
    "cost_usd": 2298
  },
  "partial": true,
  "models": {
    "deepseek-flash": 2298
  }
}
```

Known totals are partial when unknown request counts are nonzero. Reasoning tokens are not added to completion tokens. All failed and interrupted attempts are retained. Individual model distributions and step IDs are in batch.json. Scorer outputs are external evidence, not agent work.
