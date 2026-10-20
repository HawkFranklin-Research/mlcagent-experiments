# AutoML comparison on MLCA's frozen rows

Research-side comparators for R1-1 and R2-3: direct AutoGluon and FLAML, with no LLM calls. This is separate from MLCA's agent and from the cancelled reference-script arm. Supported tasks are single-label binary classification, multiclass classification (including an explicitly declared one-hot label), and regression. No experiment is launched by installation or spec generation.

## Protocol

- Reuse the last frozen split before the first test unlock. Never generate a split. Verify the MLCA event hash chain, registered source hash, membership hash, training-index hash, counts, stored train/validation hashes and frame equality, and reconstructed test plaintext hash.
- Use MLCA's table reader and interpreter for export. Dataset paths in specs resolve **relative to the experiments repository root**, independently of the working directory.
- The source must match both the spec and the registered table. A prepared-data split or a registration that changed the table (for example label normalization or column removal) is unavailable under this exporter. Do not silently undo transformations or waive a hash check.
- When `.mlca/data` was omitted from GitHub, use the controller's hash-checked `split_manifest-run-001.csv`. Missing train/validation parquet files are accepted only if rebuilding them produces the **exact original parquet byte hashes**. Different pandas/PyArrow serialization can make this unavailable: use the original MLCA environment, not a new split.
- All source rows must be covered once; train, validation and test identifiers must be unique and disjoint. Features come from the spec, not the agent's selected features. Exclude every target, leakage, identifier, evaluation-only and group column.
- Fit only on training rows, use validation for model selection/early stopping, and never refit on train plus validation. Explicitly disable AutoGluon bagging, stacking and weighted ensembles (the latter fit weights on validation labels). **Prefit AutoGluon's feature generator on training rows only**; its normal learner otherwise concatenates training and tuning rows during feature generation. Those settings are recorded with each result.
- AutoGluon retains the medium-quality default CPU candidates (including its GBM variants, CatBoost, XGBoost, Random Forest and Extra Trees), explicitly excluding NN_TORCH and FASTAI. FLAML retains its default estimator list (`auto`), recorded after resolution. Neural/PyTorch/fastai models are not installed or requested. Thread limits apply during fitting and prediction; tree models are CPU-only. Time budgets are framework search budgets, not operating-system kill deadlines. Feature-generator time is subtracted from AutoGluon's budget; an indivisible fit may overrun a framework budget. Record actual times.
- Classification predictions are derived by argmax from **one** probability prediction per evaluation partition. Save class order; score binary AUC against the declared positive label or `sorted(classes)[1]`. Absent-class multiclass AUC averages eligible per-class AUCs and lists omitted classes. AUC unavailable on a one-class evaluation set is `null`, never zero. Unsupported primary metrics fail explicitly.
- Keep failures and interrupted `started` records. A present `result.json` is never overwritten or retried. Do not delete it to improve a score.
- Optional external cohorts must match a separately registered/sealed external source, and require a logged test unlock. They enter prediction/scoring after fitting only and never participate in selection.

### MLCA saved predictions

The recorded test metrics are always retained with source `recorded`. If several distinct test metrics share the primary name, the summary marks the baseline ambiguous rather than choosing the best.

Where saved predictions have an explicit schema, the shared scorer also scores MLCA. Add this **metadata**, from inspected prediction/code provenance, to the task spec before export:

```yaml
mlca_prediction_schema:
  prediction_column: prediction  # or y_pred; omit if unavailable
  probability_columns: [proba_0, proba_1] # exact saved columns in class order
  classes: [negative_label, positive_label]
```

For regression only `prediction_column` is needed. For a binary scalar positive-class probability, `probability_columns` may be the column name as a string rather than a one-element list. Do not infer a class-to-column mapping from an unnamed matrix. The exporter copies only row ids and recognized prediction fields (`prediction`, `y_pred`, `probability`, `proba`, `proba_*`) from a hash-verified completed `audit_test` output. Unrecognized schemas are recorded as unavailable; no pickle/model is loaded. An explicit schema supports arbitrary prediction/probability column names without including feature values. Shared-score comparisons identify their source; remaining comparisons identify recorded scores.

## Environments and lock

Export uses `../mlca-revized/.venv`. AutoML has a separate environment:

```bash
uv venv --python 3.13 automl/.venv
uv pip sync --python automl/.venv/bin/python automl/requirements.lock
```

Python 3.13 is the newest supported by the chosen AutoGluon 1.5.0 release. AutoGluon 1.6.3 was considered but its Linux XGBoost extra installs `xgboost-cpu`, whereas FLAML 2.7.0's extra installs `xgboost`; both own the same Python module and resolved to different versions. 1.5.0 permits a single shared `xgboost==2.1.4`. The lock contains no torch/fastai; the XGBoost wheel has a transitive NCCL runtime dependency, but GPU training is disabled. Top-level pins and all resolved versions are in `requirements.in` and `requirements.lock`.

Regenerate the lock (resolution only, no training):

```bash
uv pip compile --python-version 3.13 automl/requirements.in -o automl/requirements.lock
```

Every result records Python, AutoGluon, FLAML, scikit-learn, NumPy, pandas and LightGBM versions, settings, CPU count, seed and fit/prediction times. AutoGluon is not guaranteed fully deterministic. Seeds are fixed for Python, NumPy, the AutoGluon learner and FLAML.

Primary implementation references: [AutoGluon installation](https://auto.gluon.ai/stable/install.html), [TabularPredictor.fit](https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.fit.html), [selected learner source](https://github.com/autogluon/autogluon/blob/v1.5.0/tabular/src/autogluon/tabular/learner/default_learner.py), [FLAML API](https://microsoft.github.io/FLAML/docs/reference/automl/automl/).

## Generate specs (metadata only)

```bash
../mlca-revized/.venv/bin/python automl/make_specs.py
```

This reads only `benchmarks/manifest.yaml`. `specs/heldout.yaml` contains Bank Marketing, Steel Plates Faults, and Wine Quality; it does not inspect those tables. Steel's seven target columns become one class label only if every row has exactly one 1. Development specs are for separately labelled smoke evidence only. Biomedical specs and their scientific choices are added at preregistration, not guessed here.

## Commands for later authorized runs

Use explicit time budgets and CPU counts agreed in the protocol; none are silently chosen by the runner. Example single pair, from the experiments root:

```bash
PYTHONPATH=../mlca-revized/src ../mlca-revized/.venv/bin/python automl/export_split.py \
  --spec automl/specs/development.yaml --task breast_cancer_diagnostic \
  --project PATH_TO_EXISTING_PROJECT \
  --work bench/automl/.work/breast_cancer_diagnostic/study/attempt-001 \
  --run-label study/attempt-001

automl/.venv/bin/python automl/run_automl.py \
  --spec automl/specs/development.yaml --task breast_cancer_diagnostic \
  --work bench/automl/.work/breast_cancer_diagnostic/study/attempt-001 \
  --framework autogluon --time-budget SECONDS --cpus CPUS \
  --out bench/automl/breast_cancer_diagnostic/study/attempt-001/autogluon
```

Then use `--framework flaml` and a separate output directory. Batch form, only after the corresponding MLCA runs exist:

```bash
PYTHONPATH=../mlca-revized/src automl/.venv/bin/python automl/run_automl.py --batch \
  --spec automl/specs/heldout.yaml --runs-root PATH_TO_MLCA_BATCH \
  --mlca-python ../mlca-revized/.venv/bin/python \
  --time-budget SECONDS --cpus CPUS --seed 42 \
  --work-root bench/automl/.work --out bench/automl
```

The exporter is a subprocess using the MLCA interpreter. Framework jobs run sequentially in the AutoML interpreter; each MLCA attempt remains a separate labelled comparison. Existing results (including failures) are skipped. The exporter retains a failed `split.json`; resuming does not silently change that evidence. The controller now preserves frozen membership outside `.mlca/data` for **future** runs; historical runs lacking both copies cannot be reconstructed.

## Outputs

Private, ignored: `bench/automl/.work/<task>/<study>/<attempt>/`: parquet partitions with original row index, `split.json`, temporary model files and export diagnostics. Model directories are deleted after scoring or failure. Do not publish private partitions or environments.

Curated layout: `bench/automl/<task>/<study>/<attempt>/<framework>/`: `result.json`, `test_predictions.parquet` (row id, true label, prediction and probability columns, **no features**), optional external predictions, and AutoGluon's cached validation `leaderboard.csv`. Cached leaderboard scores avoid another explicit validation prediction call; this is a deliberate alternative to `leaderboard(val)`.

```bash
automl/.venv/bin/python automl/summarize.py --out bench/automl
```

Writes `summary.csv` and `SUMMARY.md`, retaining failures and reporting matched-score medians and paired differences (AutoML minus MLCA). Lower RMSE is better; higher AUC is better. No failure is scored as zero. Mixed/unknown primary metrics are not subtracted.

## Validation completed during implementation

Fast synthetic integrity tests and mocked adapter contracts only. **No training smoke test, real-table AutoML run, held-out/biomedical read, MLCA command, model API call or cloud action was performed.** Local and Codespace smoke tests are intentionally deferred by the author's latest instruction. Installation/import checks do not establish that either framework completes a fit; that remains for the separately authorized smoke gate.

```bash
PYTHONPATH=.:scripts:../mlca-revized/src ../mlca-revized/.venv/bin/pytest -q tests/test_automl.py tests/test_batch_queue.py
```
