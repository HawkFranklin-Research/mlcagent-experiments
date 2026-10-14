# DRAFT preregistration — author approval required

This is preparation for evaluation, not an approved measurement protocol. No freeze or measured
run is established by this file. Smoke and offline reference acceptance remain separate.

## Arms and tasks

- Driver: MLCA Navigator and Worker, the six model IDs in `bench/models/openrouter_models_2026-10-13.csv`.
- Brain: OpenCode on the same OpenRouter models; Claude Code on the declared Haiku model;
  Codex on the declared Luna model. Client versions, flags and reported actual models retained.
- Reference: no AI, built-in actions only. Custom instructions not supported are not achieved;
  their absence cannot be interpreted as an AI failure or silently filled with hand-written code.
- Six bench tables, no custom instructions. Bio-bench S1, six S2 tasks, S3 and S4 follow STUDY.md.
- Cards in bench/cards and bio-bench/S*/task_card*.yaml; source checksums in preparation manifests.

## Settings and repetition

Proposed N=5 independent runs per worker/brain cell; one reference run per task. N remains DRAFT.
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
