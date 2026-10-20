# Work order: AutoML comparison on MLCA's frozen splits

Author decision, 2026-10-20: the AutoML comparison calls AutoGluon and FLAML directly from a small script, instead of the AMLB harness named earlier in `STUDY_DESIGN.md`. Reviewer comments: R1-1 (relevant comparison), R2-3 (comparison with existing tools).

Question answered: **on exactly the same training, validation and test rows as an MLCA run, what does a standard automated tool score, with a fixed time budget and no LLM?**

Read first: this repository's `AGENTS.md`; `scripts/run_batch.py` (`finish_job`, `prune`); `../mlca-revized/src/mlca/engine/split.py` (`freeze`); `../mlca-revized/benchmarks/manifest.yaml`.

## 1. Scope

In scope:
- Single-target **binary**, **multiclass** and **regression** tasks.
- Held-out ★ tables: `bank_marketing` (binary), `steel_plates_faults` (multiclass), `wine_quality` (regression).
- Biomedical classification and regression tasks (S1, S2 ×6, S3). Their task specs are written at the preregistration. Here, only support them: wide tables (S2: about 120 rows × 23,000 columns) and an optional external cohort (S3).
- Development tables, for the smoke test only. They are reported separately if they are reported at all.

Out of scope: multilabel, multi-output regression, clustering, representation and survival (no AutoML counterpart). Do not build the AMLB harness. Do not change MLCA (`mlca-revized`).

## 2. Hard rules

1. **Same rows as MLCA.** Partitions are rebuilt from the MLCA run's `split_manifest.csv` and verified against the hashes in its `SplitFrozen` event. Any mismatch stops that comparison with a recorded error; never fall back to a new split.
2. **No test leakage.** Fit on train only. Validation is used only for model selection and early stopping. Predict on test exactly once, after fitting. Assert that train, val and test row ids are disjoint.
3. **No refit on train + val**, because MLCA fits on training rows only: AutoGluon `refit_full` off; FLAML `retrain_full=False`.
4. **Features come from the task card, not from MLCA's choices.** Features are all columns except targets, `leakage_columns`, `identifier_columns`, `evaluation_only` and `group_column`. Record the list.
5. **Same metric as MLCA.** Use the primary metric from the run's `TaskConfigured` event, computed by one shared scorer for AutoML and MLCA (section 5).
6. **Every run is kept**, failures included. No retries to improve a score. Record the error type and message.
7. Do not open `benchmarks/heldout/` or any biomedical file during development or smoke tests. Synthetic data and one development table only.
8. Work on `master` only; no new branches. Commit locally; no push without the author's go-ahead. Never print or commit keys.

## 3. Layout

```
automl/
  WORK_ORDER.md          this file
  README.md              what it does, environments, commands, outputs
  requirements.in        top-level pins
  requirements.lock      fully pinned (uv pip compile), used on the Codespace
  specs/heldout.yaml     task specs for the 3 held-out ★ tables (generated from manifest.yaml)
  specs/development.yaml task specs for development tables (smoke tests only)
  export_split.py        runs in MLCA's environment: MLCA run -> verified parquet partitions
  run_automl.py          runs in the AutoML environment: partitions -> fitted framework -> predictions + result.json
  score.py               shared scorer (no AutoML imports; importable from both environments)
  summarize.py           results -> bench/automl/summary.csv and SUMMARY.md
  make_specs.py          manifest.yaml -> specs/*.yaml
tests/test_automl.py     fast tests (synthetic data)
bench/automl/            results (pushed); bench/automl/.work/ (ignored: partitions, models)
```

Add `bench/automl/.work/` to `.gitignore`.

## 4. Two environments

- **Export** runs in MLCA's own environment (`../mlca-revized/.venv`). It reads data with MLCA's reader (`mlca.project.inventory.read_table`), so parsing and row index are identical to the run.
- **AutoML** runs in its own environment, `automl/.venv`, created with `uv`. Use the newest Python that the chosen AutoGluon release supports; check its documentation, as it may be older than MLCA's 3.13. Install `autogluon.tabular` with the LightGBM, CatBoost and XGBoost extras only (no PyTorch or fastai models), plus `flaml[automl]`, `pandas`, `pyarrow`, `scikit-learn` and `pyyaml`. Pin everything in `requirements.lock`. Record the versions of Python, AutoGluon, FLAML, scikit-learn, NumPy, pandas and LightGBM in every `result.json`.

## 5. Components

### 5.1 Task spec (`specs/*.yaml`, made by `make_specs.py`)

One entry per task:

```yaml
- id: wine_quality
  data: ../mlca-revized/benchmarks/heldout/wine_quality_uci.csv   # path; read only at run time
  sha256: <from manifest>
  problem_type: regression            # binary | multiclass | regression
  targets: [quality]
  label_from_one_hot: false           # true when targets are one-hot columns of one class label
  excluded: []                        # leakage + identifier + evaluation_only + group_column
  external: null                      # optional {data, sha256} for a sealed external cohort (S3)
```

`steel_plates_faults` lists seven one-hot target columns with `problem_type: multiclass`. Set `label_from_one_hot: true`. The label is the name of the column holding 1. Fail if any row does not have exactly one 1. Drop all seven columns from the features.

`make_specs.py` reads `manifest.yaml`, keeps entries whose `problem_type` is binary, multiclass or regression and whose targets give one label, and writes `specs/heldout.yaml` and `specs/development.yaml`. It never reads the data files.

### 5.2 `export_split.py` (MLCA environment)

Input: a spec entry and an MLCA project directory (`.../studies/<id>/attempt-NNN/run-001`).

Steps:
1. Read `.mlca/log/events.jsonl`. Take the last `SplitFrozen` event before the first test unlock, and the last `TaskConfigured` before it. Check that the store's hash chain verifies (`mlca.log.store.Store(project).verify()`).
2. Refuse if the split's `primary_data` is a prepared artifact rather than the registered table. Check that the data file's SHA-256 equals both the spec's and the registered one.
3. Load the table with `read_table`. Read `split_manifest.csv` from the path in the event. Check:
   - its SHA-256 equals `payload["sha256"]`;
   - the training row ids, in manifest order, reproduce `payload["train_index_sha256"]` (`json.dumps(list, default=str, separators=(",", ":"))`, then SHA-256);
   - the partition counts equal `payload["counts"]`;
   - the rebuilt train and val frames equal the stored `train.parquet` and `val.parquet` (`pandas.testing.assert_frame_equal`).
4. Write `bench/automl/.work/<task>/<run-label>/{train,val,test}.parquet` (row index kept) and `split.json`, containing:
   - the run label and MLCA project path;
   - the MLCA code commit and rails version (from the run snapshot or batch state);
   - the split id, method and parameters, and every hash checked;
   - the counts and the primary metric.

   `<run-label>` is, for example, `heldout-01/bank_marketing-002`.
5. If the run never froze a split, write `split.json` with `status: no_mlca_split` and stop. That pair is reported as unavailable.

Also extract MLCA's own test result for the paired comparison:
- If the run has a completed `test_predictions.parquet` (from the `audit_test` action), copy only `row_id` and the prediction and probability columns to `.work/.../mlca_test_predictions.parquet`.
- Always also record MLCA's recorded test metric (`record_metric` with split `test`) in `split.json`, with `source: recorded`.

### 5.3 `score.py` (shared)

`score(problem_type, metric, y_true, y_pred=None, proba=None, classes=None, positive=None) -> dict`:
- **binary `roc_auc`:** the positive class is the declared positive label, or `sorted(classes)[1]` (MLCA's RL-0017 convention). `y_true` becomes an indicator for that class. Also report `balanced_accuracy`, `macro_f1` and `accuracy` from `y_pred`.
- **multiclass `roc_auc_ovr`:** the full probability matrix aligned to `classes`, `multi_class="ovr"`, `average="macro"`, `labels=classes`. If some classes are absent from the evaluated rows, use the mean of binary per-class AUCs over the classes present, and list the classes left out (MLCA's RL-0017). Also report `balanced_accuracy`, `macro_f1` and `accuracy`.
- **regression `rmse`:** also report `mae` and `r2`.
- Return `{"primary": {"name", "value"}, "secondary": {...}, "n": len(y_true), "notes": [...]}`.

Use the same scorer for MLCA's saved test predictions, when they exist, so that both sides are scored by one function.

### 5.4 `run_automl.py` (AutoML environment)

CLI: `--spec FILE --task ID --work DIR --framework {autogluon,flaml} --time-budget S --cpus N --seed 42 --out DIR`.

1. Load the partitions and `split.json`. Build features and label per section 2, rule 4, and section 5.1. Assert disjoint row ids.
2. **AutoGluon:** `TabularPredictor(label=..., problem_type=..., eval_metric=<mapped metric>, path=<.work model dir>)`, then `.fit(train, tuning_data=val, time_limit=S, presets="medium_quality", num_cpus=N)`. Keep the defaults otherwise: no `refit_full`, no bagging. Save `leaderboard(val)` as `leaderboard.csv`.
3. **FLAML:** `AutoML().fit(X_train, y_train, X_val=..., y_val=..., task=..., metric=<mapped>, time_budget=S, n_jobs=N, seed=42, eval_method="holdout", retrain_full=False)`. Save the best estimator, its configuration and its validation loss.
4. Map metric names for the installed versions, and test the mapping: binary `roc_auc`; multiclass AutoGluon `roc_auc_ovr_macro` / FLAML `roc_auc_ovr`; regression AutoGluon `root_mean_squared_error` / FLAML `rmse`.
5. Predict validation and test once each. Score both with `score.py`.
6. Write `bench/automl/<task>/<run-label>/<framework>/result.json` and `test_predictions.parquet`. The predictions file holds `row_id`, label and probabilities only, never feature values. `result.json` holds:
   - status and spec id;
   - the split hashes (copied from `split.json`);
   - the feature list and its count, and the class labels with the positive label;
   - the time budget, the actual fit and predict seconds, and the CPUs;
   - the seed, the framework settings and the versions;
   - validation and test scores, and the best model's name.
7. Delete the AutoGluon model directory after scoring; on large tables it is several GB. Set fixed seeds for Python, NumPy and the framework, and record that AutoGluon may not be fully deterministic.
8. On any exception, write `result.json` with `status: failed`, the error type and the first line of the message. Exit 0, so that a batch continues.

### 5.5 Batch driver (`run_automl.py --batch`) and `summarize.py`

- `--batch --spec specs/heldout.yaml --runs-root <MLCA batch folder>`:
  - for each task, finds the MLCA studies (`studies/<task>-NNN/attempt-*/run-001`);
  - exports each split with the MLCA interpreter, given as `--mlca-python`;
  - runs both frameworks, one job at a time, with the given budget and CPUs.
- It skips a pair that already has a `result.json`, so that a stopped batch can resume. It never deletes a failed result to rerun it.
- `summarize.py` writes:
  - `bench/automl/summary.csv`, one row per task × MLCA run × framework: MLCA test metric and its source, AutoML test metric, difference, fit seconds, status;
  - `SUMMARY.md`: a table per task, plus the median over runs.

### 5.6 Keep the split row ids with the run (controller, small change)

`split_manifest.csv` lives in `.mlca/data/`, which is never pushed. In `scripts/run_batch.py` `finish_job`, before `prune`, copy each project's `.mlca/data/splits/<split_id>/split_manifest.csv` to `<attempt>/split_manifest-<project>.csv`. It holds row ids and partition names only. Record its SHA-256 in the result. `export_split.py` uses this copy when `.mlca/data` is absent, after the same hash check. Add a test in `tests/test_batch_queue.py`.

## 6. Tests and smoke test

The coding agent completes both.

**Fast tests** (`tests/test_automl.py`, under one minute, no real data):
- Synthetic binary, multiclass (including one-hot labels) and regression tables of 300 rows. Build a minimal fake MLCA project: `events.jsonl` with `TaskConfigured` and `SplitFrozen` events whose hashes match, `split_manifest.csv`, `train.parquet` and `val.parquet`.
  - If the real `Store` hash chain makes a fake log impractical, factor the event reading so that tests can pass parsed events, and test `verify()` separately.
- The export refuses:
  - a changed manifest hash;
  - a changed data hash;
  - overlapping rows;
  - a prepared-data split;
  - a run without a split (it writes `no_mlca_split`).
- `score.py` agrees with direct scikit-learn calls for every metric, including the case of a class absent from the evaluated rows.
- `make_specs.py` output for the manifest: the held-out file contains exactly the 3 ★ tables, and `steel_plates_faults` has `label_from_one_hot: true`.

**Smoke test, local** (`automl/.venv`): both frameworks on each synthetic table with `--time-budget 20 --cpus 2`, end to end through export, run and summarize. Each `result.json` has `status: completed` and a finite test metric.

**Smoke test, Codespace** (the deployment agent runs this after review): one development table, `breast_cancer_diagnostic`, paired with one finished MLCA acceptance-02 study whose `.mlca/data` still exists, with `--time-budget 60`. Results go to `bench/automl/smoke/`, labelled as development.

## 7. Deliverables and report

- Code, tests, `README.md`, `requirements.in` and `requirements.lock`, specs, the `.gitignore` entry and the controller change in 5.6.
- One local commit on `master`.
- A short report:
  - files changed;
  - test output;
  - smoke-test `result.json` excerpts;
  - installed versions;
  - anything you could not do, with the reason.

## 8. Launch (later, deployment agent)

After the frozen MLCA held-out runs exist: `--batch --spec specs/heldout.yaml`, both frameworks, at the preregistered time budget and CPUs (proposed: 600 s and 2 CPUs, matching MLCA's 2 threads). That is about 3 tables × N MLCA runs × 2 frameworks × 10 minutes. The biomedical specs are added at the preregistration and run after the biomedical MLCA runs.
