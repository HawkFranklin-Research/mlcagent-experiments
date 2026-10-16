# Batch-02 review

Development only. Frozen code 2f04c0b (engine c2b9978, rails e433c51 / rails-0.4),
controller/evidence 0f8bc4c. Model deepseek/deepseek-v4.1-flash for all roles;
temperature 0, max output 16384, provider-default reasoning; six studies, one
large table at once. No source, rails or generation changes during active jobs.

## Outcome

Three qualifying successes: Heart, Abalone and Energy Efficiency. Each completed
through report, agreed with the declared setup and passed all external leakage
criteria. Nine failures retained; all eleven studies with a frozen split had
clean recorded fitted-column/group/time checks, including incomplete workflows.
That is not evidence of a completed final model for the failed workflows.
Eleven setups agree; Retail's static clustering omits the expected time column.
No unsupervised workflow completed, so no external hidden-label score was run.

362 requests: 793836 input +1490998 output tokens; 1369764 reasoning included
in output. Cost $1.468940104, all usage known. Cumulative since batch01:
535 requests, 1171910 input +2205700 output, cost $2.211723952.
337 requests succeeded;25 errors were output_truncated. No provider outage,
rate-limit failure or unknown request cost. Truncation is an output-budget /
generation failure, not evidence of provider unavailability.

## Shared causes and consolidated next changes

1. **Engine/template contract: per-row audits.** Breast validation audit and
Dry Bean test audit request validation_predictions.csv/test_predictions.csv.
The runtime correctly requires local Parquet with row_id and partition in
{train,val,test}. The supporting action asks for a row index but omits the exact
schema. Attempts repeat missing/invalid partition refusals. Align audit output
specifications and Worker helper documentation; preserve privacy and fit rules.
The primary score steps had completed: the failures were in supporting audits.
2. **Engine/template contract: task metric.** Covertype score_validation attempt2
recovered a frontend mismatch, but records roc_auc as its action literally asks;
the dynamic primary check correctly expects roc_auc_ovr. The validation event
therefore stayed open. Its supporting audit then hit the same row-output defect.
Make classification action/method text task-aware; never alias distinct metrics
merely to make a check pass. Model-selection confirmation remains required.
3. **Generic library/API guidance.** Adult passes unsupported pos_label to
roc_auc_score. Anuran applies unsupported multiclass-multioutput scoring, then
mixes object/integer label representations, then tries fitting during validation.
Clarify binary probability alignment and per-target categorical scoring with
training-fitted encoders reused unchanged; no fitting in evaluation.
4. **Generic feature handoff.** Bike's fitted frontend expects date-derived
columns absent in the later input. All engineered frontend operations must be
persisted in reusable importable components, not transient dataframe mutations.
5. **Output budget.** Seeds model and Optical validation stop on truncation.
25 calls hit output truncation in the batch. Keep any explicit reasoning-effort
change separate from instruction/engine revisions and record settings. Concise
script instructions alone do not control hidden reasoning.
6. **Retail: data/aim decisions.** EDA recovered an InvoiceNo aggregation error
on attempt3. Group splitting then correctly refused missing CustomerID values:
"Group split requires a complete group column". The agent's static customer
segmentation is defensible but differs from the manifest's temporal requirement.
Author question remains open; no manifest answer is inserted into the agent.
Customer-level aggregation also needs to preserve an explicit unit/row lineage;
never bypass the exact frozen-training-index fit requirement.

## Recovery and resources

Heart, Abalone and Energy include2,4,2 repair calls respectively. Full per-table
attempts/usage/time/model and the three leakage criteria are in REPORT.md and
batch.json. Covertype completed full-size registration and splitting, both model
fits, and a validation-script repair (expected54 original columns, got20 after
double preprocessing). Unlike batch01 it was not interrupted by the implementer.
Its private fingerprint database reached45033918464 bytes. This is a substantial
engine storage/time cost, not model compute. No per-study RAM cap was added.

Closed-study private fingerprint caches were archived losslessly with zstd and
round-trip uncompressed SHA256 checks, after the worker exited and external
scoring finished. Metadata: private-cache-archives.json. Restore original bytes
before privacy reads on archived projects; raw sources, steps, outputs and vaults
remain. Disposable dead-PID offline pytest scratch (326 folders) was cleaned to
provide archive headroom; gate logs/XML remain in the external cache. This did
not delete any scientific input, model, script, result or event log. No study was
paused, interrupted, resumed or rerun due to storage in this batch.

## Evidence and limitations

curated/MANIFEST.csv hashes the copied text evidence; OMITTED.csv hashes binary
models/tables retained only in raw attempts. Runtime environment requirements
lockfiles are included. Private data/env/vault/cache and local notes excluded.
No hand-written analysis, held-out access, biomedical run, paper edit, push,
freeze, tag, version bump or publication. Model/rails fixed within the batch.
AcceptanceK3 and rails ablation have not started.
