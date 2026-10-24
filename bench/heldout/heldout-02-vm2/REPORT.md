# Heldout batch 2

Model, provider, runtime source and rail setting fixed for every study in this batch.
Comparison plan: None
Acceptance outcome and rail effect are reported separately; failed runs are retained.

| Table / replicate | Completed | Setup agrees | Leakage: columns / groups / time | Success | Furthest event / step | Category | Repairs | Requests | Input / output / reasoning | Known cost USD | Wall s | Reported models |
|---|---|---|---|---|---|---|---:|---:|---|---:|---:|---|
| steel_plates_faults / 1 | True | False | True / True / True | False | report / 12_report_generate_reports_P7T3Q5AE | setup_disagreement | 4 | 40 | 119481 / 96662 / 79039 | None | 623.6369247436523 | {"deepseek-flash": 40} |
| steel_plates_faults / 2 | False | False | None / True / True | False | data / 02_data_count_target_XJ47YXG2 | test_lock | 2 | 7 | 23087 / 8203 / 6275 | None | 68.88082122802734 | {"deepseek-flash": 7} |
| steel_plates_faults / 3 | True | False | True / True / True | False | report / 12_report_generate_reports_FS46MVHQ | setup_disagreement | 2 | 38 | 105321 / 78043 / 63939 | None | 521.8889651298523 | {"deepseek-flash": 38} |
| wine_quality / 1 | True | False | True / True / True | False | report / 12_report_generate_reports_27EMAMSV | setup_disagreement | 1 | 38 | 102887 / 77527 / 63293 | None | 545.4423584938049 | {"deepseek-flash": 38} |
| wine_quality / 2 | True | False | True / True / True | False | report / 13_report_generate_reports_BKWENCAF | setup_disagreement | 0 | 39 | 97249 / 65437 / 52213 | None | 486.08821177482605 | {"deepseek-flash": 39} |
| wine_quality / 3 | True | False | True / True / True | False | report / 12_report_generate_reports_880GSZMM | setup_disagreement | 0 | 37 | 95297 / 65427 / 52214 | None | 453.6555151939392 | {"deepseek-flash": 37} |

## Environment preflight

Last check 2026-10-24T14:47:16.972561+00:00 on mlca-runner-2.us-central1-a.c.pelliscope-scout.internal, Python 3.13.17: default 3e677b448fb0 (matches reference), lifelines 53f3d70c0a00 (matches reference)

## Batch accounting

```json
{
  "requests": 199,
  "known": {
    "input_tokens": 543322,
    "output_tokens": 391299,
    "reasoning_tokens": 316973,
    "cost_usd": null
  },
  "unknown": {
    "input_tokens": 0,
    "output_tokens": 0,
    "cost_usd": 199
  },
  "partial": true,
  "models": {
    "deepseek-flash": 199
  }
}
```

## Running accounting since batch-01

```json
{
  "requests": 642,
  "known": {
    "input_tokens": 1700947,
    "output_tokens": 1384184,
    "reasoning_tokens": 1134157,
    "cost_usd": null
  },
  "unknown": {
    "input_tokens": 0,
    "output_tokens": 0,
    "cost_usd": 642
  },
  "partial": true,
  "models": {
    "deepseek-flash": 642
  }
}
```

Known totals are partial when unknown request counts are nonzero. Reasoning tokens are not added to completion tokens. All failed and interrupted attempts are retained. Individual model distributions and step IDs are in batch.json. Scorer outputs are external evidence, not agent work.
