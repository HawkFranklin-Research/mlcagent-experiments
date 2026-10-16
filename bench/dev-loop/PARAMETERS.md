# Rails development parameters

Work order 04 Part B, authorized again by the author on 16 October 2026.
Development evidence only; no freeze or held-out/biomedical evaluation.

- Twelve development tables only, in manifest simple-to-complex order. The agent
  receives no manifest, expected task, targets or traps. Only evaluation-only
  column names are passed to registration through `exclude_columns`.
- Driver mode; Navigator and Worker: `openrouter/free`; temperature 0;
  maximum output 16,384 tokens; request timeout 90 seconds; 16 requests/minute.
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
