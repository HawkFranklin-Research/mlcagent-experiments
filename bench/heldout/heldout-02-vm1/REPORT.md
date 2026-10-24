# Heldout batch 2

Model, provider, runtime source and rail setting fixed for every study in this batch.
Comparison plan: None
Acceptance outcome and rail effect are reported separately; failed runs are retained.

| Table / replicate | Completed | Setup agrees | Leakage: columns / groups / time | Success | Furthest event / step | Category | Repairs | Requests | Input / output / reasoning | Known cost USD | Wall s | Reported models |
|---|---|---|---|---|---|---|---:|---:|---|---:|---:|---|
| bank_marketing / 1 | True | True | True / True / True | True | report / 12_report_generate_reports_5ZCEX3EH |  | 1 | 38 | 102827 / 72850 / 59054 | None | 587.8816883563995 | {"deepseek-flash": 38} |
| bank_marketing / 2 | True | True | True / True / True | True | report / 12_report_generate_reports_6S6FAHQW |  | 3 | 41 | 116516 / 107190 / 90369 | None | 748.6365261077881 | {"deepseek-flash": 41} |
| bank_marketing / 3 | True | True | True / True / True | True | report / 12_report_generate_reports_G77DGDX6 |  | 4 | 41 | 115119 / 93590 / 77849 | None | 685.1159372329712 | {"deepseek-flash": 41} |
| wholesale_customers / 1 | True | True | True / True / True | True | report / 12_report_generate_reports_MNPMQP25 |  | 1 | 37 | 95528 / 67443 / 53666 | None | 474.2056918144226 | {"deepseek-flash": 37} |
| wholesale_customers / 2 | True | True | True / True / True | True | report / 12_report_generate_reports_V16GWEY5 |  | 1 | 40 | 107449 / 102250 / 86014 | None | 641.0188500881195 | {"deepseek-flash": 40} |
| wholesale_customers / 3 | True | True | True / True / True | True | report / 12_report_generate_reports_6CKT52AB |  | 2 | 39 | 107257 / 102311 / 84755 | None | 643.4208300113678 | {"deepseek-flash": 39} |

## Environment preflight

Last check 2026-10-24T14:46:42.873084+00:00 on mlca-eval-runner.us-central1-a.c.pelliscope-scout.internal, Python 3.13.17: default 3e677b448fb0 (matches reference), lifelines 53f3d70c0a00 (matches reference)

## Batch accounting

```json
{
  "requests": 236,
  "known": {
    "input_tokens": 644696,
    "output_tokens": 545634,
    "reasoning_tokens": 451707,
    "cost_usd": null
  },
  "unknown": {
    "input_tokens": 0,
    "output_tokens": 0,
    "cost_usd": 236
  },
  "partial": true,
  "models": {
    "deepseek-flash": 236
  }
}
```

## Running accounting since batch-01

```json
{
  "requests": 679,
  "known": {
    "input_tokens": 1802321,
    "output_tokens": 1538519,
    "reasoning_tokens": 1268891,
    "cost_usd": null
  },
  "unknown": {
    "input_tokens": 0,
    "output_tokens": 0,
    "cost_usd": 679
  },
  "partial": true,
  "models": {
    "deepseek-flash": 679
  }
}
```

Known totals are partial when unknown request counts are nonzero. Reasoning tokens are not added to completion tokens. All failed and interrupted attempts are retained. Individual model distributions and step IDs are in batch.json. Scorer outputs are external evidence, not agent work.
