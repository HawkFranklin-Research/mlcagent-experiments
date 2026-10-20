# Heldout batch 1

Model, provider, runtime source and rail setting fixed for every study in this batch.
Comparison plan: None
Acceptance outcome and rail effect are reported separately; failed runs are retained.

| Table / replicate | Completed | Setup agrees | Leakage: columns / groups / time | Success | Furthest event / step | Category | Repairs | Requests | Input / output / reasoning | Known cost USD | Wall s | Reported models |
|---|---|---|---|---|---|---|---:|---:|---|---:|---:|---|
| bank_marketing / 1 | False | True | True / True / True | False | validate / 10_validate_score_validation_DNJF8B36 | runtime_error:ValueError | 2 | 29 | 78144 / 88550 / 74136 | None | 506.86422324180603 | {"deepseek-flash": 29} |
| bank_marketing / 2 | True | True | True / True / True | True | report / 14_report_generate_reports_8DN7PQBD |  | 0 | 42 | 105433 / 82638 / 67265 | None | 540.8084211349487 | {"deepseek-flash": 42} |
| bank_marketing / 3 | True | True | True / True / True | True | report / 12_report_generate_reports_0YZ4QVNV |  | 0 | 37 | 95712 / 115992 / 100730 | None | 671.2165961265564 | {"deepseek-flash": 37} |
| steel_plates_faults / 1 | True | False | True / True / True | False | report / 12_report_generate_reports_Z1KWY872 | setup_disagreement | 1 | 37 | 103536 / 84873 / 68086 | None | 469.70223116874695 | {"deepseek-flash": 37} |
| steel_plates_faults / 2 | True | False | True / True / True | False | report / 12_report_generate_reports_Z5NYAEGT | setup_disagreement | 0 | 37 | 101127 / 72516 / 59881 | None | 425.21464252471924 | {"deepseek-flash": 37} |
| steel_plates_faults / 3 | True | False | True / True / True | False | report / 13_report_generate_reports_WMC62RXH | setup_disagreement | 0 | 39 | 100760 / 84733 / 70579 | None | 486.9649238586426 | {"deepseek-flash": 39} |
| wine_quality / 1 | True | False | True / True / True | False | report / 13_report_generate_reports_JPE2GS51 | setup_disagreement | 0 | 40 | 100879 / 68149 / 54712 | None | 416.02307653427124 | {"deepseek-flash": 40} |
| wine_quality / 2 | True | False | True / True / True | False | report / 13_report_generate_reports_3Q5MHVKF | setup_disagreement | 1 | 42 | 107855 / 87552 / 72604 | None | 506.4344217777252 | {"deepseek-flash": 42} |
| wine_quality / 3 | False | False | True / True / True | False | validate / 09_validate_score_validation_MAS434FE | gate_blocked | 1 | 27 | 65255 / 51487 / 41835 | None | 296.1573462486267 | {"deepseek-flash": 27} |
| wholesale_customers / 1 | True | True | True / True / True | True | report / 12_report_generate_reports_70KXVAHX |  | 1 | 37 | 95664 / 97453 / 81465 | None | 544.6173164844513 | {"deepseek-flash": 37} |
| wholesale_customers / 2 | True | True | True / True / True | True | report / 12_report_generate_reports_9T6T82NW |  | 1 | 37 | 97032 / 76349 / 60729 | None | 433.7443513870239 | {"deepseek-flash": 37} |
| wholesale_customers / 3 | True | True | True / True / True | True | report / 12_report_generate_reports_13CQ99C2 |  | 2 | 39 | 106228 / 82593 / 65162 | None | 454.050185918808 | {"deepseek-flash": 39} |

## Batch accounting

```json
{
  "requests": 443,
  "known": {
    "input_tokens": 1157625,
    "output_tokens": 992885,
    "reasoning_tokens": 817184,
    "cost_usd": null
  },
  "unknown": {
    "input_tokens": 0,
    "output_tokens": 0,
    "cost_usd": 443
  },
  "partial": true,
  "models": {
    "deepseek-flash": 443
  }
}
```

## Running accounting since batch-01

```json
{
  "requests": 443,
  "known": {
    "input_tokens": 1157625,
    "output_tokens": 992885,
    "reasoning_tokens": 817184,
    "cost_usd": null
  },
  "unknown": {
    "input_tokens": 0,
    "output_tokens": 0,
    "cost_usd": 443
  },
  "partial": true,
  "models": {
    "deepseek-flash": 443
  }
}
```

Known totals are partial when unknown request counts are nonzero. Reasoning tokens are not added to completion tokens. All failed and interrupted attempts are retained. Individual model distributions and step IDs are in batch.json. Scorer outputs are external evidence, not agent work.
