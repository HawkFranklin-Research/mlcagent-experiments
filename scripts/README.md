# Experiment preparation and execution

Use the code repository's environment; these scripts make no model calls during preparation.

1. Set `SOURCE_DATA` in git-ignored `paths.local.yaml`.
2. Run `python bio-bench/S*/prepare_data.py` for each study. S1 records a source checksum only;
   S2–S4 write deterministic tables and input/output hashes under git-ignored `data_prep/`.
3. `python scripts/run_card.py <card> --out <fresh-directory>` resolves local paths. It writes
   a git-ignored resolved card alongside the run.
4. `mlca bio-bench export <run-output> --to <fresh-export-directory>` curates results.
5. `python scripts/summarize.py <exports-root>` builds cross-arm tables.

The cards and PREREGISTRATION.md are DRAFT, not authorization for measured or live runs. Finalize
scientific placeholders before measured runs. Record both original and resolved task-card hashes.
Source inputs are never modified. Preparation does not fit preprocessing or select predictors.
Within-study repeated splits remain custom instructions; no E3 built-in is introduced.

## External scoring (work order 04)

`score_inference.py` compares recorded setup decisions with researcher metadata.
It reports exact metadata agreement, not proof that other scientific aims are invalid.

`score_unsupervised.py` joins original hidden labels to local per-row outputs by
row identifier after the MLCA run. Clustering uses test ARI/NMI. Representation
uses a fixed external linear probe fitted on training embeddings and labels and
scored on test embeddings. The outputs explicitly identify this as external
scoring, not agent work. Hidden labels must never be registered with MLCA.

Use fresh `--out` paths; the scorers never overwrite a result or launch an agent.

## Development batches (work order 05)

This author-approved development loop is separate from frozen measurements.
From a clean committed code/controller tree, with OPENROUTER_API_KEY available:

```bash
../mlca-revized/.venv/bin/python scripts/run_batch.py \
  --batch 1 --out bench/dev-loop/batch-01 \
  --model deepseek/deepseek-v4.1-flash --concurrency 6
```

The queue uses development entries only, six concurrent fresh studies, one large
table at a time, shared pacing and one cost ledger. It never sends researcher
expectations or leakage fields to the agent; only approved evaluation-column
exclusions cross the boundary. The Worker writes every analysis script.

Resume with the identical source commits using `--out <same-folder> --resume`.
Finished studies are never repeated; interrupted attempts stay preserved.
`REPORT.md`, `batch.json` and `cost-ledger.json` are continuously written. Raw
projects stay local; curate before committing any run evidence.

External leakage scoring checks fitted column names, group separation and time
ordering; it is a researcher measurement, not agent work. Missing evidence stays
unavailable. All three checks plus setup agreement and completion define success.
`--replicates 3` requires prior successes for every table. `--rails-off` requires
successful acceptance on the same model and code; engine safeguards stay active.

Parameters, caps and scientific boundaries: `bench/dev-loop/PARAMETERS.md` and
the code repository's `docs/prompts/05_batch_development_loop.prompt.md`.

### Local monitoring utilities and reproducibility

`scripts/codespace_keepalive.sh` is a machine-specific cloud monitor, not study
runtime code. That exact path is ignored and left local. The controller's
reproducibility check remains unchanged: modified tracked files and other
untracked files under `scripts/`, or the agent's `src/`, `templates/` and
`rails/`, still block a run. Commit authorized controller changes before the
smoke-test agent tries a fresh output directory.

### Local Ollama smoke diagnostics

A short API probe does not exercise the size or duration of MLCA requests. The
controller previously fixed the SDK timeout at 90 seconds and the outer provider
call deadline at 300 seconds. New Ollama batches now default to a 300-second SDK timeout, a 600-second hard
call deadline and one concurrent study. Other providers keep 90/300 seconds and
six concurrent studies. Explicit flags override defaults for a new batch; existing
batches and comparison plans keep their recorded values. New batches can set
`--sdk-timeout-s` and `--model-call-deadline-s`. Both values
are recorded, passed into the frozen provider configuration, and immutable on
resume. Shared comparison plans also fix these settings: CLI overrides cannot
silently change them.

For the next separately executed **single-table** Ollama diagnostic, a candidate
configuration is `--sdk-timeout-s 300 --model-call-deadline-s 600 --concurrency 1`.
This is a diagnostic setting, not a verified cure or a change to existing measured
runs. It does not add provider retries or alter scripts, rails, model settings,
sandbox limits, or disk gates. A timeout can still occur at the longer deadline.

Every job's `result.provider_errors` in `batch.json` now retains request ID,
error type/category, HTTP status when available, latency, pacing wait and rate-limit
wait count. Request bodies, response headers and credentials are not copied.
Keep this payload-free failure summary when removing temporary smoke artifacts;
generic provider errors cannot be diagnosed from a completion count alone.

Disk checks use the output filesystem. On the inspected local machine `/tmp` is
a roughly 7.3-GiB RAM filesystem, so it cannot satisfy the 16-GiB large-table gate
even when empty. Increasing provider timeouts cannot address that. Large-table
smokes require an approved **disk-backed scratch location outside the repository**
with at least 16 GiB free at launch. The smoke owner must revise the earlier
`/tmp`-only restriction before testing large tables; do not redirect outputs or
relax the disk gate silently. No cloud compute is required.

### Ready-to-use local Ollama configuration

`bash scripts/run_ollama_local.sh --dry-run` prints the configured command without
reading credentials, creating output or making requests. The smoke-test agent can
start **one** Breast Cancer study with `bash scripts/run_ollama_local.sh`.

The launcher selects MiniMax M3.1 Flash, structured outputs on, low reasoning,
300/600-second SDK/hard-call timeouts, concurrency 1, two threads and 20 RPM.
It creates a fresh disk-backed session under `~/.cache/mlca-smoke-ollama/`, outside
the repository and `/tmp`. This implements the author's request to configure local
execution after `/tmp` proved unsuitable. It uses `OLLAMA_API_KEY` if already set,
otherwise loads the designated key file into the shell environment only. No key
is printed or passed in arguments. It does not access cloud compute, resume an
old batch, launch the twelve-table queue, or delete failure evidence. Existing
sandbox and disk preflights still apply. The longer timeouts need live smoke
verification; generic provider errors may have another cause.
