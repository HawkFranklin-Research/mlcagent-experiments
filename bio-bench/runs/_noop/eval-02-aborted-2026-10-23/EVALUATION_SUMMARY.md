# Biomedical evaluation, arm A — eval-02 interrupted run record

Protocol: frozen MLCA `freeze-v2` (commit `7dc0aedfb674e102f3cc96a92931a1a382eaaea0`), rails `0.15`, direct DeepSeek `deepseek-flash`, reasoning effort `low`; 9 approved tasks × 3 planned replicates. The batch was intentionally stopped after an oversized setup request was detected. This is a partial operational record, not a completed evaluation or a valid estimate of the planned success rate. No run was resumed or retried.

## Results by replicate

| Task / replicate | Completed | Furthest step | Failure/interruption category | Setup agrees | Leakage clean | Repairs | Test metric | Test predictions saved |
|---|---:|---|---|---:|---:|---:|---|---:|
| `s1_anemia-001` | yes | report | — | True | True | 0 | See run artifacts | yes |
| `s1_anemia-002` | yes | report | — | True | True | 1 | See run artifacts | yes |
| `s1_anemia-003` | no | test | interrupted_by_author | True | True | 0 | — | no |
| `s2_rna_er-001` | no | onboarding/data | interrupted_by_author | True | unknown (not reached) | 0 | — | no |
| `s2_rna_er-002` | not started | — | queued when batch stopped | — | — | — | — | — |
| `s2_rna_er-003` | not started | — | queued when batch stopped | — | — | — | — | — |
| `s2_rna_pr-001` | not started | — | queued when batch stopped | — | — | — | — | — |
| `s2_rna_pr-002` | not started | — | queued when batch stopped | — | — | — | — | — |
| `s2_rna_pr-003` | not started | — | queued when batch stopped | — | — | — | — | — |
| `s2_rna_her2-001` | not started | — | queued when batch stopped | — | — | — | — | — |
| `s2_rna_her2-002` | not started | — | queued when batch stopped | — | — | — | — | — |
| `s2_rna_her2-003` | not started | — | queued when batch stopped | — | — | — | — | — |
| `s2_proteome_er-001` | not started | — | queued when batch stopped | — | — | — | — | — |
| `s2_proteome_er-002` | not started | — | queued when batch stopped | — | — | — | — | — |
| `s2_proteome_er-003` | not started | — | queued when batch stopped | — | — | — | — | — |
| `s2_proteome_pr-001` | not started | — | queued when batch stopped | — | — | — | — | — |
| `s2_proteome_pr-002` | not started | — | queued when batch stopped | — | — | — | — | — |
| `s2_proteome_pr-003` | not started | — | queued when batch stopped | — | — | — | — | — |
| `s2_proteome_her2-001` | not started | — | queued when batch stopped | — | — | — | — | — |
| `s2_proteome_her2-002` | not started | — | queued when batch stopped | — | — | — | — | — |
| `s2_proteome_her2-003` | not started | — | queued when batch stopped | — | — | — | — | — |
| `s3_pam50-001` | not started | — | queued when batch stopped | — | — | — | — | — |
| `s3_pam50-002` | not started | — | queued when batch stopped | — | — | — | — | — |
| `s3_pam50-003` | not started | — | queued when batch stopped | — | — | — | — | — |
| `s4_hnscc-001` | not started | — | queued when batch stopped | — | — | — | — | — |
| `s4_hnscc-002` | not started | — | queued when batch stopped | — | — | — | — | — |
| `s4_hnscc-003` | not started | — | queued when batch stopped | — | — | — | — | — |

## Totals and interpretation

- Planned: 27 studies. Completed: 2/27; both completed S1 anemia studies had setup agreement and clean leakage checks. Two further studies started and were interrupted; 23 were never started.
- The 2/27 (7.4%) completed share is only a raw count from the partial batch. Because the batch was deliberately stopped before execution of the plan, it is not a valid evaluation success rate and must not be compared with the preregistered 50% reopening threshold or the author's 80% expectation.
- Requests: 109 total; model counts: `deepseek-flash` 108, `unavailable` 1.
- Known tokens: input 677157, output 208717, reasoning 170260. Cost is unavailable; token usage is partial where the provider did not return usage fields.
- S4 setup rescoring: not applicable; no S4 study started.
- Test predictions were saved for both completed S1 studies. The interrupted studies did not save test predictions. No test metric is summarized here; consult the curated run artifacts for completed-run metrics.

## Failure and stop causes

| Run(s) | Recorded cause | Interpretation |
|---|---|---|
| `s2_rna_er-001` | `interrupted_by_author` during onboarding/data | The setup request egress record reports 1,063,982 bytes sent, exceeding the 200 kB per-request cap specified by E-W1. This was the major issue that caused the batch to be stopped. No model/workflow failure conclusion is drawn from this interruption. |
| `s1_anemia-003` | `interrupted_by_author` at test | This concurrently active study was stopped when the batch was halted over the oversized setup request; it is not classified as a model failure. |
| Remaining 23 jobs | queued; not started | No outcome can be assigned. |

## Comparison with eval-01

Eval-01 completed all 27 attempts but only `s1_anemia-001` completed the workflow (1/3 for S1; 0/3 completed for each other task). Eval-02 stopped early, so its entries below are counts observed before the stop, not comparable replicate estimates.

| Task | Eval-01 completed | Eval-02 completed before stop | Eval-02 planned runs not executed |
|---|---:|---:|---:|
| S1 anemia | 1/3 | 2/3 | 1 interrupted |
| S2 RNA ER | 0/3 | 0/3 | 1 interrupted, 2 not started |
| S2 RNA PR | 0/3 | Not evaluated | 3 not started |
| S2 RNA HER2 | 0/3 | Not evaluated | 3 not started |
| S2 proteome ER | 0/3 | Not evaluated | 3 not started |
| S2 proteome PR | 0/3 | Not evaluated | 3 not started |
| S2 proteome HER2 | 0/3 | Not evaluated | 3 not started |
| S3 PAM50 | 0/3 | Not evaluated | 3 not started |
| S4 HNSCC | 0/3 | Not evaluated | 3 not started |

The eval-01 S4 setup score was later corrected to compare survival duration/event fields; that correction does not change its 0/3 workflow completions. Eval-02 had no S4 run to rescore.
