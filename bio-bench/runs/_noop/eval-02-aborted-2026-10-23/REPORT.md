# Evaluation batch 1

Model, provider, runtime source and rail setting fixed for every study in this batch.
Comparison plan: None
Acceptance outcome and rail effect are reported separately; failed runs are retained.

| Table / replicate | Completed | Setup agrees | Leakage: columns / groups / time | Success | Furthest event / step | Category | Repairs | Requests | Input / output / reasoning | Known cost USD | Wall s | Reported models |
|---|---|---|---|---|---|---|---:|---:|---|---:|---:|---|
| s1_anemia / 1 | True | True | True / True / True | True | report / 12_report_generate_reports_FZKMVHTT |  | 0 | 36 | 89543 / 77028 / 63437 | None | 496.3988447189331 | {"deepseek-flash": 36} |
| s1_anemia / 2 | True | True | True / True / True | True | report / 13_report_generate_reports_P7TYHWNN |  | 1 | 40 | 98971 / 78708 / 63954 | None | 501.02506399154663 | {"deepseek-flash": 40} |
| s1_anemia / 3 | False | True | True / True / True | False | test / 10_test_score_test_3T5EGKKF | interrupted_by_author | 0 | 30 | 73587 / 52336 / 42661 | None | 343.6919710636139 | {"deepseek-flash": 29, "unavailable": 1} |
| s2_rna_er / 1 | False | True | None / True / True | False | data / 01_onboard_confirm_brief_805PNBXT | interrupted_by_author | 0 | 3 | 415056 / 645 / 208 | None | 339.72377276420593 | {"deepseek-flash": 3} |
| s2_rna_er / 2 | queued | None | None / None / None | None | setup / None |  | None | None | None / None / None | None | None | {} |
| s2_rna_er / 3 | queued | None | None / None / None | None | setup / None |  | None | None | None / None / None | None | None | {} |
| s2_rna_pr / 1 | queued | None | None / None / None | None | setup / None |  | None | None | None / None / None | None | None | {} |
| s2_rna_pr / 2 | queued | None | None / None / None | None | setup / None |  | None | None | None / None / None | None | None | {} |
| s2_rna_pr / 3 | queued | None | None / None / None | None | setup / None |  | None | None | None / None / None | None | None | {} |
| s2_rna_her2 / 1 | queued | None | None / None / None | None | setup / None |  | None | None | None / None / None | None | None | {} |
| s2_rna_her2 / 2 | queued | None | None / None / None | None | setup / None |  | None | None | None / None / None | None | None | {} |
| s2_rna_her2 / 3 | queued | None | None / None / None | None | setup / None |  | None | None | None / None / None | None | None | {} |
| s2_proteome_er / 1 | queued | None | None / None / None | None | setup / None |  | None | None | None / None / None | None | None | {} |
| s2_proteome_er / 2 | queued | None | None / None / None | None | setup / None |  | None | None | None / None / None | None | None | {} |
| s2_proteome_er / 3 | queued | None | None / None / None | None | setup / None |  | None | None | None / None / None | None | None | {} |
| s2_proteome_pr / 1 | queued | None | None / None / None | None | setup / None |  | None | None | None / None / None | None | None | {} |
| s2_proteome_pr / 2 | queued | None | None / None / None | None | setup / None |  | None | None | None / None / None | None | None | {} |
| s2_proteome_pr / 3 | queued | None | None / None / None | None | setup / None |  | None | None | None / None / None | None | None | {} |
| s2_proteome_her2 / 1 | queued | None | None / None / None | None | setup / None |  | None | None | None / None / None | None | None | {} |
| s2_proteome_her2 / 2 | queued | None | None / None / None | None | setup / None |  | None | None | None / None / None | None | None | {} |
| s2_proteome_her2 / 3 | queued | None | None / None / None | None | setup / None |  | None | None | None / None / None | None | None | {} |
| s3_pam50 / 1 | queued | None | None / None / None | None | setup / None |  | None | None | None / None / None | None | None | {} |
| s3_pam50 / 2 | queued | None | None / None / None | None | setup / None |  | None | None | None / None / None | None | None | {} |
| s3_pam50 / 3 | queued | None | None / None / None | None | setup / None |  | None | None | None / None / None | None | None | {} |
| s4_hnscc / 1 | queued | None | None / None / None | None | setup / None |  | None | None | None / None / None | None | None | {} |
| s4_hnscc / 2 | queued | None | None / None / None | None | setup / None |  | None | None | None / None / None | None | None | {} |
| s4_hnscc / 3 | queued | None | None / None / None | None | setup / None |  | None | None | None / None / None | None | None | {} |

## Batch accounting

```json
{
  "requests": 109,
  "known": {
    "input_tokens": 677157,
    "output_tokens": 208717,
    "reasoning_tokens": 170260,
    "cost_usd": null
  },
  "unknown": {
    "input_tokens": 1,
    "output_tokens": 1,
    "cost_usd": 109
  },
  "partial": true,
  "models": {
    "deepseek-flash": 108,
    "unavailable": 1
  }
}
```

## Running accounting since batch-01

```json
{
  "requests": 291,
  "known": {
    "input_tokens": 1281573,
    "output_tokens": 688716,
    "reasoning_tokens": 577267,
    "cost_usd": null
  },
  "unknown": {
    "input_tokens": 22,
    "output_tokens": 22,
    "cost_usd": 291
  },
  "partial": true,
  "models": {
    "unavailable": 22,
    "deepseek-flash": 269
  }
}
```

Known totals are partial when unknown request counts are nonzero. Reasoning tokens are not added to completion tokens. All failed and interrupted attempts are retained. Individual model distributions and step IDs are in batch.json. Scorer outputs are external evidence, not agent work.
