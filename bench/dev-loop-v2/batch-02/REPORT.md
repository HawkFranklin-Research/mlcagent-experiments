# Development batch 2

Model, provider, runtime source and rail setting fixed for every study in this batch.
Comparison plan: None
Acceptance outcome and rail effect are reported separately; failed runs are retained.

| Table / replicate | Completed | Setup agrees | Leakage: columns / groups / time | Success | Furthest event / step | Category | Repairs | Requests | Input / output / reasoning | Known cost USD | Wall s | Reported models |
|---|---|---|---|---|---|---|---:|---:|---|---:|---:|---|
| gli85_glioma / 1 | True | True | True / True / True | True | report / 12_report_generate_reports_5MXABFST |  | 1 | 38 | 105198 / 99367 / 82738 | None | 1087.9003858566284 | {"deepseek-flash": 38} |
| gli85_glioma / 2 | True | True | True / True / True | True | report / 12_report_generate_reports_3QGTPA33 |  | 0 | 37 | 98369 / 92605 / 78479 | None | 1068.363665819168 | {"deepseek-flash": 37} |
| gli85_glioma / 3 | True | True | True / True / True | True | report / 12_report_generate_reports_16EBZZSD |  | 0 | 37 | 98067 / 101839 / 87268 | None | 1069.5494108200073 | {"deepseek-flash": 37} |
| smk187_airway / 1 | True | True | True / True / True | True | report / 12_report_generate_reports_82WEMNRY |  | 0 | 36 | 94591 / 77940 / 63936 | None | 1095.7456214427948 | {"deepseek-flash": 36} |
| smk187_airway / 2 | True | True | True / True / True | True | report / 13_report_generate_reports_Y944PPV3 |  | 3 | 42 | 112737 / 89241 / 71712 | None | 1096.7896962165833 | {"deepseek-flash": 42} |
| smk187_airway / 3 | True | True | True / True / True | True | report / 12_report_generate_reports_V9CAFQ16 |  | 1 | 37 | 98568 / 92882 / 77890 | None | 1114.2040677070618 | {"deepseek-flash": 37} |
| gcm_tumour_types / 1 | True | True | True / True / True | True | report / 12_report_generate_reports_15ZM0WWA |  | 2 | 38 | 105583 / 95814 / 81069 | None | 1189.6441173553467 | {"deepseek-flash": 38} |
| gcm_tumour_types / 2 | True | True | True / True / True | True | report / 12_report_generate_reports_0VB7ESH3 |  | 2 | 39 | 109371 / 125516 / 109666 | None | 1593.7973256111145 | {"deepseek-flash": 39} |
| gcm_tumour_types / 3 | True | True | True / True / True | True | report / 13_report_generate_reports_7BVX9RZH |  | 1 | 40 | 103144 / 75283 / 61810 | None | 1242.840139389038 | {"deepseek-flash": 40} |

## Batch accounting

```json
{
  "requests": 344,
  "known": {
    "input_tokens": 925628,
    "output_tokens": 850487,
    "reasoning_tokens": 714568,
    "cost_usd": null
  },
  "unknown": {
    "input_tokens": 0,
    "output_tokens": 0,
    "cost_usd": 344
  },
  "partial": true,
  "models": {
    "deepseek-flash": 344
  }
}
```

## Running accounting since batch-01

```json
{
  "requests": 880,
  "known": {
    "input_tokens": 2405084,
    "output_tokens": 2218896,
    "reasoning_tokens": 1862809,
    "cost_usd": null
  },
  "unknown": {
    "input_tokens": 0,
    "output_tokens": 0,
    "cost_usd": 880
  },
  "partial": true,
  "models": {
    "deepseek-flash": 880
  }
}
```

Known totals are partial when unknown request counts are nonzero. Reasoning tokens are not added to completion tokens. All failed and interrupted attempts are retained. Individual model distributions and step IDs are in batch.json. Scorer outputs are external evidence, not agent work.

## Testing agent closeout

All nine requested replicates reached the final report event. Each run completed with exact setup agreement and clean leakage checks, so each table meets the pooled acceptance rule (3 of 3; required 2 of 3). No failure category was recorded. No engine, rail, or controller changes are recommended from this batch.

| Development table | Successful runs | Acceptance |
|---|---:|---|
| gli85_glioma | 3/3 | Accepted |
| smk187_airway | 3/3 | Accepted |
| gcm_tumour_types | 3/3 | Accepted |

Batch totals: 344 requests; 925,628 input tokens, 850,487 output tokens, and 714,568 reasoning tokens. Provider cost was unavailable.
