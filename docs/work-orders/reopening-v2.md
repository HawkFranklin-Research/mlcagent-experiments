# Work order: reopening after the biomedical evaluation (rule 7) → rails-0.14, freeze-v2, arm A again

Repositories: `../mlca-revized` (engine, rails, development tables) and `mlcagent-experiments` (this repository: controller, runs). Branch `master` only in both; no new branches. Commit locally; do not push or upload.

## Why

The biomedical evaluation, arm A, on `freeze-v1` (`a9f4e35`, rails-0.13, `deepseek-flash`) succeeded in 1/27 runs (`bio-bench/runs/deepseek-flash/eval-01/`). That is above the 50% failure threshold, so the design is reopened once (STUDY_DESIGN.md §4 item 7, author 2026-10-20):
- the development set is extended by about three public tables, never a held-out or biomedical table;
- rails are added under the same ledger rules;
- the new version is accepted and frozen;
- the biomedical studies are run again.

The first evaluation is reported in full and is never rerun or edited. The second version is reported as a later version whose development was informed by the first evaluation's failures.

## Read first

1. `bio-bench/runs/deepseek-flash/eval-01/EVALUATION_SUMMARY.md`, especially "Failure causes" and "Deviation: S4 setup scoring".
2. STUDY_DESIGN.md §3–4 (read only): `/home/prime/Documents/github/v-papers/ml_copilot_paper/ES/submission2/experiments/STUDY_DESIGN.md`.
3. `../mlca-revized/rails/LEDGER.md`, `../mlca-revized/docs/rails.md`, `../mlca-revized/docs/prompts/05_batch_development_loop.prompt.md` (its loop and rules apply unless changed here).
4. `PREREGISTRATION.md` § "Phase 3: reopening".

## Hard rules

1. **Biomedical data stays closed.** Do not open `data_prep/` or any biomedical table. You may read the eval-01 run records (logs, scripts, egress and usage records) to understand failures. Rails and engine fixes must be generic. They must not contain biomedical or development dataset names, file names or column names (the existing rails regression test checks this).
2. **Held-out tables** (`benchmarks/heldout/`) are not used.
3. **Survival stays outside the rails** (author, 2026-10-15). No survival rail and no survival development table. An engine bug that shows up on a survival task may be fixed as an engine bug, with a test that uses synthetic data.
4. Development tables are never sampled by the controller.
5. No retries to improve an outcome. Every failed run is kept and reported.
6. **Keys:** `DEEPSEEK_API_KEY` from the environment only; never print, log or commit it.

## Step 1 · Development tables (already added; verify)

Three wide gene-expression tables from OpenML, so that the development set covers tables with tens of thousands of columns:
- none comes from TCGA, CPTAC or any biomedical source used in the evaluation;
- none is a held-out table.

| id | OpenML | Rows × features | Task | Source |
|---|---|---|---|---|
| `gli85_glioma` | 45089 (GLI) | 85 × 22,283 | binary | Freije et al., Cancer Res 2004 |
| `smk187_airway` | 45100 (SMK) | 187 × 19,993 | binary | Spira et al., Nat Med 2007 |
| `gcm_tumour_types` | 1106 (GCM) | 190 × 16,063 | multiclass (14) | Ramaswamy et al., PNAS 2001 |

The tables are in `../mlca-revized/benchmarks/datasets/`, with entries in `benchmarks/manifest.yaml` (role `development`) and `SHA256SUMS.txt`. Verify the checksums before the first batch. If the files are missing:
- download them with `benchmarks/datasets/pull_datasets.py` or the OpenML API (ARFF → CSV, unchanged values, class column last);
- add the manifest entries in the existing format (`leakage_columns: []`, `identifier_columns: []`);
- commit before any batch.

## Step 2 · Engine fixes (one commit each, regression test, ledger entry `B-00NN` citing eval-01)

**E-W1, bounded setup inventory for wide tables** (cause of 21/27 failures).
- `inventory()` (`src/mlca/autonomous/inference.py`) sends one summary per column, so a 23,123-column table made a 6.2 MB request.
- When a table has more than 200 columns, list in full:
  - every non-numeric column;
  - every column with at most 20 distinct values;
  - every column named in the user brief;
  - the first and last 20 columns.
- Summarise the remaining numeric columns as one block per dtype, giving:
  - the count;
  - the total missing values and the number of columns with any missing value;
  - the range of their minima, maxima and means;
  - 20 example names.
- Give the total column count.
- Validation of the proposal keeps using the full table, so any existing column can still be named.
- The egress ledger records exactly what was sent.
- Apply the same bound wherever a schema or column list reaches a model:
  - the Navigator's state view;
  - Worker work orders and briefs;
  - repair prompts.
- Test with a synthetic table of 25,000 numeric columns plus a 2-level and a 5-level label. Every request MLCA builds through the end of the workflow must stay under 200 kB, and the label columns must appear in the setup inventory.

**E-W2, the provider's error message is recorded.**
- On an HTTP error, record the provider's error type and message in `usage.jsonl` and the failure event, shortened to 500 characters.
- Never record request content or headers.
- Test with a stubbed 400 response.

**E-M1, the primary-metric recovery names the exact key.**
- Validation stopped when the Worker recorded only per-model names (`roc_auc:logistic`).
- The scoring-only retry's instruction must:
  - name the exact metric key and split it expects (for example `record_metric("roc_auc", value, split="val")` for the selected model);
  - say that per-model names are additional and do not count.
- Test: a validation script that records only per-model names gets a retry instruction containing the exact key.

**E-S1, the survival data review** (S4-003). "Choose the outcome column before reviewing the table" (`src/mlca/engine/data_review.py:75`) was raised after the split, although the setup named duration and event.
- Find the cause in the run record.
- If it is an engine bug, fix it and add a test with synthetic survival data.
- If it is not an engine bug, record the finding in the ledger and change nothing.

## Step 3 · Rails (rails-0.14)

- Add rails only for generic Worker or Navigator failures, each linked to evidence in the ledger.
- Starting candidate, **RL-0028 (instruction, Worker, validation):** record the primary metric of the selected model under its exact name (for example `roc_auc`, split `val`), in addition to any per-model values.
- Add further rails only for failures seen in the development batches below.
- Bump `rails/VERSION` to `rails-0.14`.

## Step 4 · Development batches (`bench/dev-loop-v2/`)

- **Settings:**
  - `--provider deepseek --model deepseek-flash --reasoning-effort low --max-output-tokens 32768 --rpm 60`;
  - `--concurrency 6 --threads 2`;
  - per-study deadline 5400 s.
  These match the held-out and biomedical runs.
- **Batch-01:** all 15 development tables once, on the version with Steps 2–3. It shows whether the 12 accepted tables still pass after the changes; the controller requires all tables in batch-01.
- **Then:** classify the failures together, make one consolidated change and rerun the failed tables, as in work order 05.
- **Acceptance (pooled 2-of-3 rule, STUDY_DESIGN §3 item 6):**
  - Each new table, and each old table that failed in this series, runs 3 times on one unchanged version and must succeed in at least 2.
  - Old tables that pass batch-01 keep their earlier acceptance.
  - A run succeeds when it completes, its setup agrees, and it is leakage-clean.
- **Caps:** 6 batches or 3 wall-clock days. At a cap, stop and report.
- **Ablation:** none.

After each batch, report in one short message:
- the batch table;
- tokens;
- the changes you will make.

## Step 5 · Freeze

- Full code suite (about 20 minutes) and ruff on the changed files.
- Update `docs/HANDOFF.md`.
- Tag `freeze-v2` on the accepted commit; local tag, no push.
- Record the commit, rails version and digest in `PREREGISTRATION.md` § "Phase 3".

## Step 6 · Biomedical evaluation, arm A, second version

- Same 9 tasks, briefs, data revision (`data_prep/hf-111de16`, checksum-verified by the controller), model and settings as eval-01.
- New folder: `bio-bench/runs/deepseek-flash/eval-02`.

```bash
git -C ../mlca-revized checkout freeze-v2
python scripts/run_batch.py --role evaluation --data-root data_prep/hf-111de16 \
  --out bio-bench/runs/deepseek-flash/eval-02 --batch 1 --model deepseek-flash --provider deepseek \
  --reasoning-effort low --replicates 3 --concurrency 2 --threads 2 --max-output-tokens 32768 --rpm 60
git -C ../mlca-revized checkout master
```

- Write `EVALUATION_SUMMARY.md` in the same format as eval-01, including the failure causes.
- Commit curated, as for eval-01: no row-level Parquet, split-membership CSVs or control files.

## Deliverables

- Commits in both repositories:
  - tables;
  - E-W1, E-W2, E-M1, E-S1;
  - rails-0.14;
  - batch evidence;
  - the freeze record;
  - eval-02.
- A final report:
  - per-table acceptance and the version each table was accepted on;
  - tokens per batch and in total;
  - eval-02's per-task results;
  - the success count against rule 7 (author's expectation: at least 80%).
