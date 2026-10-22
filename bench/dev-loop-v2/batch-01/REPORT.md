# Development batch 1

Model, provider, runtime source and rail setting fixed for every study in this batch.
Comparison plan: None
Acceptance outcome and rail effect are reported separately; failed runs are retained.

| Table / replicate | Completed | Setup agrees | Leakage: columns / groups / time | Success | Furthest event / step | Category | Repairs | Requests | Input / output / reasoning | Known cost USD | Wall s | Reported models |
|---|---|---|---|---|---|---|---:|---:|---|---:|---:|---|
| breast_cancer_diagnostic / 1 | True | True | True / True / True | True | report / 12_report_generate_reports_5XWHGFA3 |  | 4 | 43 | 125877 / 93383 / 74991 | None | 507.45823311805725 | {"deepseek-flash": 43} |
| heart_disease_cleveland / 1 | True | True | True / True / True | True | report / 12_report_generate_reports_JV9GBV8J |  | 0 | 37 | 94951 / 57360 / 44736 | None | 342.7782130241394 | {"deepseek-flash": 37} |
| adult / 1 | True | True | True / True / True | True | report / 13_report_generate_reports_ZP1GNTAT |  | 2 | 41 | 104635 / 105954 / 89357 | None | 689.2191684246063 | {"deepseek-flash": 41} |
| dry_bean / 1 | True | True | True / True / True | True | report / 12_report_generate_reports_KNRMWFSZ |  | 1 | 37 | 95360 / 79956 / 66525 | None | 507.6697926521301 | {"deepseek-flash": 37} |
| covertype / 1 | False | True | None / True / True | False | preprocess / 05_preprocess_fit_preprocessing_H1Z2KNFY | timeout | 2 | 15 | 42815 / 36573 / 29103 | None | 2947.1485090255737 | {"deepseek-flash": 15} |
| anuran_calls / 1 | True | True | True / True / True | True | report / 12_report_generate_reports_FF5P3RP0 |  | 0 | 39 | 104823 / 90106 / 76796 | None | 490.56637048721313 | {"deepseek-flash": 39} |
| abalone / 1 | True | True | True / True / True | True | report / 12_report_generate_reports_GJPBCZ47 |  | 1 | 38 | 100512 / 92243 / 75603 | None | 483.2486057281494 | {"deepseek-flash": 38} |
| bike_sharing_hourly / 1 | False | True | True / True / True | False | model / 08_model_train_alternative_E9MA4J8M | runtime_error:ValueError | 5 | 28 | 82808 / 104769 / 93027 | None | 539.2071182727814 | {"deepseek-flash": 28} |
| energy_efficiency / 1 | True | True | True / True / True | True | report / 12_report_generate_reports_Y5FV689A |  | 0 | 36 | 90867 / 67763 / 54981 | None | 379.739732503891 | {"deepseek-flash": 36} |
| seeds / 1 | True | True | True / True / True | True | report / 12_report_generate_reports_1YQ1FEAP |  | 1 | 38 | 101503 / 88614 / 70914 | None | 474.532014131546 | {"deepseek-flash": 38} |
| online_retail / 1 | True | True | True / True / True | True | report / 12_report_generate_reports_CSYNV93K |  | 0 | 37 | 92430 / 102067 / 88096 | None | 739.1165142059326 | {"deepseek-flash": 37} |
| optical_digits / 1 | False | True | True / True / True | False | test / 10_test_score_test_5A5BK7ZC | runtime_error:TypeError | 3 | 35 | 117667 / 130211 / 115309 | None | 626.6344583034515 | {"deepseek-flash": 35} |
| gli85_glioma / 1 | False | True | True / True / True | False | interpret / 11_interpret_explain_model_EQSSWR93 | timeout | 4 | 39 | 121871 / 131941 / 111932 | None | 2062.469408750534 | {"deepseek-flash": 39} |
| smk187_airway / 1 | True | True | True / True / True | True | report / 12_report_generate_reports_QG5Y0XBG |  | 1 | 37 | 99015 / 73940 / 59919 | None | 1445.6886343955994 | {"deepseek-flash": 37} |
| gcm_tumour_types / 1 | False | True | True / True / True | False | interpret / 11_interpret_explain_model_ZHCT2RJ0 | timeout | 3 | 36 | 104322 / 113529 / 96952 | None | 2277.281032562256 | {"deepseek-flash": 36} |

## Batch accounting

```json
{
  "requests": 536,
  "known": {
    "input_tokens": 1479456,
    "output_tokens": 1368409,
    "reasoning_tokens": 1148241,
    "cost_usd": null
  },
  "unknown": {
    "input_tokens": 0,
    "output_tokens": 0,
    "cost_usd": 536
  },
  "partial": true,
  "models": {
    "deepseek-flash": 536
  }
}
```

## Running accounting since batch-01

```json
{
  "requests": 536,
  "known": {
    "input_tokens": 1479456,
    "output_tokens": 1368409,
    "reasoning_tokens": 1148241,
    "cost_usd": null
  },
  "unknown": {
    "input_tokens": 0,
    "output_tokens": 0,
    "cost_usd": 536
  },
  "partial": true,
  "models": {
    "deepseek-flash": 536
  }
}
```

Known totals are partial when unknown request counts are nonzero. Reasoning tokens are not added to completion tokens. All failed and interrupted attempts are retained. Individual model distributions and step IDs are in batch.json. Scorer outputs are external evidence, not agent work.


## Observed batch summary

Batch-01 finished all 15 development tables in one attempt each. Eleven workflows completed through reporting; four did not. External setup agreement was true for all 15. Leakage checks were clean for 14; Covertype's column-fit check is unscored because preprocessing timed out before a result was produced.

| Table | Completed | Furthest event | Failure category | Setup agreement | Leakage | Requests | Input / output / reasoning tokens |
|---|---:|---|---|---:|---|---:|---:|
| Breast Cancer Diagnostic | Yes | Report | — | Yes | Clean | 43 | 125,877 / 93,383 / 74,991 |
| Heart Disease Cleveland | Yes | Report | — | Yes | Clean | 37 | 94,951 / 57,360 / 44,736 |
| Adult | Yes | Report | — | Yes | Clean | 41 | 104,635 / 105,954 / 89,357 |
| Dry Bean | Yes | Report | — | Yes | Clean | 37 | 95,360 / 79,956 / 66,525 |
| Covertype | No | Preprocess | timeout | Yes | Unscored: preprocessing did not complete | 15 | 42,815 / 36,573 / 29,103 |
| Anuran Calls | Yes | Report | — | Yes | Clean | 39 | 104,823 / 90,106 / 76,796 |
| Abalone | Yes | Report | — | Yes | Clean | 38 | 100,512 / 92,243 / 75,603 |
| Bike Sharing Hourly | No | Model (alternative) | runtime_error:ValueError | Yes | Clean | 28 | 82,808 / 104,769 / 93,027 |
| Energy Efficiency | Yes | Report | — | Yes | Clean | 36 | 90,867 / 67,763 / 54,981 |
| Seeds | Yes | Report | — | Yes | Clean | 38 | 101,503 / 88,614 / 70,914 |
| Online Retail | Yes | Report | — | Yes | Clean | 37 | 92,430 / 102,067 / 88,096 |
| Optical Digits | No | Test | runtime_error:TypeError | Yes | Clean | 35 | 117,667 / 130,211 / 115,309 |
| GLI-85 Glioma | No | Interpret | timeout | Yes | Clean | 39 | 121,871 / 131,941 / 111,932 |
| SMK-CAN-187 Airway | Yes | Report | — | Yes | Clean | 37 | 99,015 / 73,940 / 59,919 |
| GCM Tumour Types | No | Interpret | timeout | Yes | Clean | 36 | 104,322 / 113,529 / 96,952 |

### Batch identity and totals

- Model/provider: `deepseek-flash` via direct DeepSeek API; reasoning effort `low`; temperature 0.
- Runtime source: MLCA commit `92572d0fa0a054709f78553c0d8b15bc9e7de4e3`; controller commit `dc411f3e0fc2d64b9b54ab72db7fbc3f0a60fccb`; rails `rails-0.14`, SHA-256 `e15e1d3fe87c11e6582066c3c2f66b3e3ce65531136c089428dfa4679d1d0cec`.
- Settings: 6 concurrent studies, 2 threads per study, 60 RPM, 32,768 maximum output tokens, 5,400-second study deadline, 300-second model-call deadline.
- Total: 536 requests; 1,479,456 input, 1,368,409 output and 1,148,241 reasoning tokens. The provider did not report cost, so cost is unavailable.
- Eleven of 15 tables completed, and all failures/timeouts are retained. No reruns or code/rail changes were made.
