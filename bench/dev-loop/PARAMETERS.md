# Rails development parameters

Work order 04 Part B, authorized again by the author on 16 October 2026.
Development evidence only; no freeze or held-out/biomedical evaluation.

- Twelve development tables only, in manifest simple-to-complex order. The agent
  receives no manifest, expected task, targets or traps. Only evaluation-only
  column names are passed to registration through `exclude_columns`.
- Driver mode; Navigator and Worker: `openrouter/free`; temperature 0;
  maximum output 16,384 tokens; SDK I/O timeout 90 seconds (not a strict wall-clock deadline); 16 requests/minute.
  No Inspector, paid model, alternate provider or model fallback list.
- Fresh projects, autonomous window zero, existing default engine/script limits,
  three Worker attempts, bounded provider cooldowns. Sandbox A is required.
- K=3 consecutive fresh completions per table on unchanged rails. Maximum twelve
  rounds and five wall-clock days. Every failed attempt is retained.
- Round 1 runs every table once. Subsequent development rounds rerun failures
  and at least half the passes. Acceptance runs all tables three times unchanged.
- Before each table, check the real free-request allowance. Initial planning
  estimate: 50 requests/table; replace with the observed mean when it is higher.
  If remaining allowance cannot carry the remaining round, stop and report.
- Initial account check: authenticated; 0/1,000 free requests used, 1,000 remaining.
  This was a metadata request, not inference. Refresh before actual execution.
- Changes must be generic and failure-linked. Engine bugs remain separate from
  rails. No changing a scientific safeguard to force a completion.
- `round-NN/<dataset>/run-NNN/` preserves original projects. `curated/` holds
  hashed, allowlisted exports for local commits; raw projects remain local.
- Inference metadata agreement and any hidden-label scoring are external
  evaluators, not agent work. Completion is reported separately from agreement.
- No pushes, uploads, package version changes, publication or freeze.

## Implementing source and offline gates

Part A source: `32530ae` (A3 `fb8abf2`, A4 `6c7d3aa`). Full code suite:
380 passed, 1 skipped in 1189.07 s; native sandbox A; Ruff passes. The controller
and external scorers passed four focused offline tests. Source/head and rails
must stay unchanged within each round. No reference analysis is substituted for
Worker code. Automatic CLI sign-in remains a later usability task.

## Active: work order 05 (16 October 2026)

This section supersedes the earlier round parameters above. Rounds 1–2 remain
exploratory evidence and are not compared against the new baseline.

- Driver mode, Navigator and Worker: `deepseek/deepseek-v4.1-flash` via OpenRouter,
  temperature 0, max output 16,384 tokens; no model substitutions or fallbacks.
  Structured outputs explicitly enabled after the official catalogue advertised
  response_format and structured_outputs for this exact model.
- Paid calls authorized with no spending ceiling. Actual known costs, unknown
  request counts and partial totals are reported per batch and cumulatively.
- All twelve development tables in batch-01; subsequent batches failed tables
  only. Six parallel studies, one request limiter shared across all processes;
  initial pacing ceiling 60 requests/minute (a local setting, not a provider quota).
- At most one table over 500,000 rows at once. Tables remain full size; any
  agent-selected sampling is its recorded decision. Numerical threads per study
  are available cores divided by concurrency (minimum one).
- Per study deadline 5,400 s; hard model-call deadline 300 s; SDK I/O timeout 90 s;
  default script deadline 300 s; three Worker attempts. Sandbox A required.
- No additional per-study memory ceiling is enforced. Machine RAM and CPU counts
  are captured in batch.json; available RAM, swap, memory pressure and free disk
  are sampled in resources.jsonl every thirty seconds. Concurrency is not raised
  without pressure evidence.
- Each study has isolated configuration and a fresh project; resume preserves
  interrupted attempts and never reruns finished studies. Source commits, model,
  rails digest and parameters are fixed within the batch.
- Success requires workflow completion, setup agreement and all three external
  leakage checks. Unknown evidence is unavailable, not clean.
- After every table succeeds once: K=3 fresh successes per table on one unchanged
  version (36 studies). Then all twelve once with instruction rails disabled
  (rails-0.0), same model and engine. No fixes based on the ablation.
- Cap: ten batches or five wall-clock days from batch-01 launch.
- Generic fixes are consolidated between batches, with engine and rail changes
  in separate commits and ledger entries. Held-out tables remain untouched.

Queue: `scripts/run_batch.py`; external scorers: `scripts/score_inference.py` and
`scripts/score_leakage.py`. The queue's offline tests use tiny process stubs, not
no-AI analysis runs. Four queue tests and two leakage tests pass. Shared engine
controls commit: c448e56. Full native suite remains the Step 1.7 launch gate.

## Batch03 preparation after batch02

Batch02 evidence commit4a9a60d:3 qualifying successes (Heart, Abalone, Energy),
9 retained failures. Engine/template contracts and instruction rails are repaired
between batches, with no model or generation-setting change. Model remains
DeepSeek V4.1 Flash, temperature0,max output16384, provider-default reasoning.
The final full native gate must pass before launching batch03.

Batch03 queues eight failed tables: breast_cancer_diagnostic, adult, dry_bean,
covertype, anuran_calls, bike_sharing_hourly, seeds, optical_digits. The ninth
failure, Online Retail, is temporarily deferred pending the author's choice of
static customer segmentation versus the existing temporal criterion. No scoring
expectation or previous result is rewritten. If decided before launch, record
the prospective decision and update this scope before any live request.

The existing entity-preparation action and prepares_data handoff remain the
path for changing analysis units before the frozen split. No synthetic or
hand-written reference analysis is substituted. Completed studies are not rerun
until acceptance. Six studies in parallel; one large atonce; same pacing,
limits, full-size tables and sandboxA. Cache storage costs and byte-verified
archival are reported separately from model compute and model repairs.
