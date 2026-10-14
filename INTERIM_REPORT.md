# MLCA v2 — interim local acceptance report

Snapshot: **14 October 2026, 03:26:17 UTC (08:56:17 IST)**. WP7 remains active; the main agent continues the offline acceptance jobs. This report records development evidence and does not establish a code freeze or measured AI-run results.

Completed WP1–6 work is present in local Git history:

| Work package | Repository / commit | Completed scope |
|---|---|---|
| WP1 | Code `829f2f1` | Sealed external inputs and ordered bio-bench execution |
| WP2 | Code `f720622` | Outcome-dependent review on training rows and concrete errors |
| WP3 | Code `bbc72fd` | Isolated brain-mode benchmarks and client usage capture |
| WP4 | Code `2ea2506` | Curated run exports and precise custom-instruction outcomes |
| WP5 | Experiments `f85fc4d` | Deterministic cohort preparation and draft evaluation material |
| WP6 | Code `cd498a3` | Implementation and MCP status documentation |

WP7 includes an **uncommitted questions-performance fix and two regression tests**. Read-only diff inspection confirms those changes in the code working tree. The main agent reports the final full offline suite as **296 passed, 1 skipped in 1146.03 seconds**; PowerShell validation is unavailable. This reporting pass did not rerun tests. Existing implementation-status prose predates these WP7 results and still needs final reconciliation.

Seven no-AI reference development acceptance attempts were launched: S1 and six S2 receptor tasks. Final result JSON currently exists for five tasks and records **zero model requests** for each; the two active tasks have no final result JSON yet. The main agent reports zero model calls across the batch. These attempts exercise deterministic reference execution; they provide no measured AI orchestration or live-client validation.

| Task | Interim outcome |
|---|---|
| S1 anemia | Core workflow completed; all four custom instructions recorded as **not achieved**. Overall status is `not_achieved`, not full protocol success. |
| S2 proteome ER | Failed at interpretation: **timeout**. |
| S2 proteome PR | Failed at interpretation: **timeout**. |
| S2 proteome HER2 | Failed at test scoring: bootstrap produced NaN and strict JSON serialization failed (`ValueError`). |
| S2 RNA ER | Failed at interpretation: **timeout**, confirmed by the `StepFailed` event category. |
| S2 RNA HER2 | Still active; final outcome pending. |
| S2 RNA PR | Still active; final outcome pending. |

Evidence inspected: local commit history and working-tree status, the final batch's result JSON, curated benchmark summaries, and failure events. Curated outputs are under `bio-bench/*/runs/reference/2026-10-14-final/`. One proteome ER development attempt was interrupted and resumed; its result explicitly records that resumption. Failure and intervention logs are retained, and no success has been fabricated. Export counts indicate files exist; hash and exclusion auditing remains unfinished.

Remaining work: finish the active jobs; audit exports, hashes and exclusions; reconcile every outcome and limitation into the acceptance documentation; and prepare final WP7 local commits and documentation under the existing authorization. The main agent owns the auditor and acceptance documents. Live client/model validation, author approvals, final preregistration and a future measurement freeze remain outstanding. No scientific acceptance or readiness for measured runs is asserted.
