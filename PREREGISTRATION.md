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


## Phase 2: biomedical evaluation, arm A (2026-10-22, author-approved)

- freeze freeze-v1 = a9f4e35, rails-0.13; agent run in autonomous mode (`mlca run`), no task card;
- data: Hugging Face `HawkFranklin-Research/mlcagent-bio-bench` revision `111de16fe082d4beab8ca70e779601b156c596c8`, SHA-256 verified;
- tasks: the 9 tasks in `bio-bench/manifest.yaml` (S1; S2 ER, PR, HER2 × RNA, proteome; S3; S4), all `approved: true`;
- starting briefs: `bio-bench/briefs/<task id>.md`, written before any run, kept verbatim with word counts;
- model deepseek-flash (direct DeepSeek API), reasoning effort low, temperature 0, max output tokens 32768;
- 3 runs per task, concurrency 2, threads 2, sandbox A, 5400 s deadline per study; no reruns to improve an outcome;
- external cohorts (S3, S4) registered sealed: unseen during setup, opened by the engine only after the test unlock;
- a run succeeds when the workflow completes, its setup agrees with the manifest, and the required leakage checks hold (STUDY_DESIGN.md, item 5); reopening rule: STUDY_DESIGN.md rule 7.

Author decisions (2026-10-22), each following the submitted study and its code:
- S1: Hemoglobin stays a feature. The label's dependence on it is reported as a finding, not removed.
- S2 HER2: ERBB2.Proteogenomic.Status, else Her2.Updated.Clinical.Status, else HER2.Amplified (15 positives).
- S3: Normal stays in the training labels; it is absent in CPTAC and excluded from the external metrics.
- S4: problem type survival (overall survival months and status), CPTAC as the sealed external cohort; survival is outside the rails.

Arm B (human on the loop, `--halt-window 60`) uses the same tasks, briefs and model, one study at a time. **Not run in this revision** (author, 2026-10-23): arm B and closed models are future work, run only if reviewers ask after this revision.

Result (2026-10-22): `bio-bench/runs/deepseek-flash/eval-01/`, 1/27 runs successful; rule 7 applies.
Deviation, S4 setup scoring: the reference `targets: []` could never agree, because MLCA requires a
non-empty target list for survival. Survival setup is now scored on problem type, duration and event;
original and re-scored files are both kept (EVALUATION_SUMMARY.md, "Deviation"). The success count is unchanged.

## Phase 3: reopening (rule 7), second version (2026-10-22, author-approved)

- Trigger: eval-01, 26/27 runs unsuccessful (threshold 50%). eval-01 is reported in full, never rerun or edited.
- Development set extended by three public wide gene-expression tables from OpenML (none from TCGA, CPTAC or a
  held-out table): GLI-85 (45089), SMK-CAN-187 (45100), GCM (1106). Survival stays outside the rails.
- Changes, linked to eval-01 failures in the ledger: bounded setup inventory for wide tables; provider error
  messages recorded; exact primary-metric key in the validation recovery; survival data-review check
  investigated as a possible engine bug; rails-0.14 for generic Worker failures.
- Development: deepseek-flash (direct DeepSeek), reasoning effort low; batch-01 on all 15 development tables;
  pooled 2-of-3 acceptance; caps 6 batches or 3 days; no ablation.
- Freeze (2026-10-23): tag `freeze-v2` = `7dc0aedfb674e102f3cc96a92931a1a382eaaea0`, rails-0.15, rails SHA-256
  `4fe378883a4cd7f8aa6b626dc5943d441a382d89e0be3081714bc0023debf03b`. Acceptance: the 12 v1 tables keep theirs
  (v2 batch-01 regression check, 9/12); GLI-85, SMK-CAN-187 and GCM 3/3 each in v2 batch-02 (`bench/dev-loop-v2/batch-02`).
  Full offline suite on the tagged commit: 545 passed, 1 skipped.
- Evaluation: eval-02 repeats eval-01 exactly (tasks, briefs, data revision, model, settings, 3 runs) on freeze-v2.
  The second version is reported as a later version whose development was informed by eval-01's failures.
- Stopping rule (author, 2026-10-23): the 12 v1 tables keep their acceptance (batch-01 is their regression check); the 3 wide
  tables get at most one fix round after batch-02; eval-02 is the last agent run of this revision, whatever its result.
- Work order: `docs/work-orders/reopening-v2.md`.

## Safeguard test (2026-10-24, author-approved; added after eval-02 had started, before any safeguard result)

- Question (R1-5): which flaws the frozen engine refuses, which only warn, which only the external scorer catches, and which go undetected.
- Method: flawed analysis scripts (and a correct control for each) submitted through MLCA's step runner at `freeze-v2` with the scripted fake provider; no model calls, no agent runs, no change to MLCA. The flaw list (`safeguard-tests/flaws.yaml`) is committed before the first run and not changed after seeing outcomes.
- Not an agent evaluation: it does not affect the stopping rule (eval-02 remains the last agent run).
- Work order: `safeguard-tests/WORK_ORDER.md`.
- Run (2026-10-24): `safeguard-tests/runs/2026-10-24-official/` (commit `199fb5df`); one amendment before any flaw was run (Heart Disease scripts impute missing values), recorded in `safeguard-tests/README.md`.

## AutoML comparison on the biomedical tasks (2026-10-24, author-approved; after eval-02, before any biomedical AutoML result)

- Question (R1-1, R2-3): on the same frozen splits as each eval-02 MLCA run, how do AutoGluon 1.5.0 and FLAML 2.7.0 compare on the primary test metric (ROC AUC; one-vs-rest ROC AUC for S3), and at what time?
- Tasks: S1 anemia, S2 receptor tasks (6), S3 PAM50. S4 HNSCC (survival) has no AutoML counterpart and is not compared. Features: every column except targets and identifier/leakage/evaluation-only/group columns (S1 keeps Hemoglobin, as for MLCA).
- Settings as the held-out comparison: 600 s search budget per framework, 2 CPUs, seed 42; AutoGluon medium quality without bagging, stacking, weighted ensemble or neural models, feature generator fitted on training rows only; FLAML default estimators. No LLM calls.
- Every MLCA attempt is reported; attempts without a frozen split are export failures, not scores. No result is rerun. Spec: `automl/specs/biomedical.yaml`; protocol `automl/README.md`.

