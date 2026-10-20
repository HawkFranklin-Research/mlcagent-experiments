# DRAFT preregistration — author approval required

This is preparation for evaluation, not an approved measurement protocol. No freeze or measured
run is established by this file. Smoke runs remain separate. There is no no-AI reference arm.

## Arms and tasks

- Driver: MLCA Navigator and Worker, the six model IDs in `bench/models/openrouter_models_2026-10-13.csv`.
- Brain: OpenCode on the same OpenRouter models; Claude Code on the declared Haiku model;
  Codex on the declared Luna model. Client versions, flags and reported actual models retained.
- Six bench tables, no custom instructions. Bio-bench S1, six S2 tasks, S3 and S4 follow STUDY.md.
- Cards in bench/cards and bio-bench/S*/task_card*.yaml; source checksums in preparation manifests.

## Settings and repetition

Proposed N=5 independent runs per worker/brain cell. N remains DRAFT.
Seed 42 for the declared split, 60/20/20; train-fitted preprocessing, validation-only selection
and thresholds, one test unlock, sealed external cohort. Runtime dependencies and template hashes
are frozen per project. Final code commit and model catalogue hash must be recorded before runs.
Draft budgets: $1 per run, $30 per card batch; client costs may be unavailable and brain-mode
monetary limits are not independently enforced by MLCA. Client wall limit: 1800 seconds.
No fallback models. Dropped/unavailable models, failures and retries are retained and reported.

## Policy and metrics

Confirm declared study/validation sub-goals and unlock the test once; no undeclared packages.
Property-triggered questions follow their recorded recommendation with explicit policy answers,
never counted as human decisions. Custom instructions are issued once in declared order, with
before/after timing in the card. Preserve every exact text, word count, scope decision and outcome.
Report completion and not-achieved steps; failure categories, recovery, interventions, time,
tokens, costs, egress, provenance and predictive metrics with exact/derived/client-reported/
unavailable labels. Unknown is never zero. Curated exports are checksum verified.

## Outstanding scientific choices

Finalize S3 harmonisation/Normal-class handling and S4 clustering rule/repeats before measured
runs. Placeholder wording in draft instructions is not permission for an agent to invent choices.
S1 source citation and label definition remain author provenance items. Raw inputs are unchanged.
Neither this draft nor preparatory manifests are scientific results or journal-ready claims.

## Phase 1b: held-out tables (2026-10-20)

- freeze freeze-v1 = a9f4e35, rails-0.13;
- model deepseek-flash (direct DeepSeek API), reasoning effort low, temperature 0, max output tokens 32768;
- 3 runs per table, concurrency 2, threads 2, sandbox A, 5400 s deadline per study;
- tables: bank_marketing, steel_plates_faults, wine_quality, wholesale_customers;
- AutoML: AutoGluon 1.5.0 and FLAML 2.7.0, 600 s and 2 CPUs per framework per run, seed 42, on the 3 ★ tables (not wholesale_customers, which is clustering);
- comparison metrics: binary ROC-AUC, multiclass one-vs-rest macro ROC-AUC, regression RMSE.

