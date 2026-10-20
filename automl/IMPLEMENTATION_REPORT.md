# AutoML implementation report — 20 October 2026

## Delivered

- Manifest-only task-spec generator and checked-in development/held-out specifications (exactly the three eligible held-out tasks; no data files opened).
- MLCA-environment split exporter: event-chain verification; registered-source, membership, training-index, partition and test plaintext hashes; row uniqueness/disjointness/counts; exact stored frame equality; hash-checked public manifest fallback; explicit unavailable/error results. Prepared-data and ambiguous branched histories fail closed.
- Direct AutoGluon/FLAML adapters, shared binary/multiclass/regression scorer, per-row predictions, optional post-unlock external scoring, framework metadata, deterministic seed settings and cleanup.
- Sequential batch/resume controller and CSV/Markdown summaries, preserving every attempt and failure. Shared rescoring of agent predictions requires an explicit probability/class schema; otherwise recorded scores retain their source and ambiguous scores are not subtracted.
- Small `scripts/run_batch.py` change: atomically preserve each selected frozen row-id manifest before pruning and record its SHA-256. Retention errors are recorded separately from measured MLCA outcomes.
- Two-environment instructions, fully pinned requirements, ignored private working data and environments, and synthetic tests.

## Actual checks

```
PYTHONPATH=.:scripts:../mlca-revized/src ../mlca-revized/.venv/bin/pytest -q tests \
  --ignore=tests/test_material.py --ignore=tests/test_unsupervised_scorer.py
78 passed in 3.16s
```

The two excluded existing files respectively read biomedical material and fit a linear probe. New tests use synthetic 300-row tables or mocked framework calls: no actual estimator fitting. Coverage includes changed source/manifest/train/test hashes, overlap, training-index/count errors, a prepared-data split, missing split, original-byte reconstruction after private data removal, first-unlock selection, one-hot labels, shared scores, probability alignment, missing classes, framework train/validation/refit settings, prediction call counts, cleanup, resume and failure summaries.

- Ruff checks passed for `automl/`, `scripts/run_batch.py`, `tests/test_automl.py` and `tests/test_batch_queue.py`.
- `git diff --check` passed.
- Both installed framework imports succeeded, without model construction/fitting.
- AutoGluon's three configured metric names resolve successfully in the installed release.
- `uv pip check --python automl/.venv/bin/python`: all 46 installed packages compatible.

## Installed environment

| Component | Version |
|---|---|
| Python | 3.13.12 |
| AutoGluon Tabular | 1.5.0 |
| FLAML | 2.7.0 |
| scikit-learn | 1.7.2 |
| NumPy | 2.3.5 |
| pandas | 2.3.3 |
| PyArrow | 20.0.0 |
| LightGBM | 4.6.0 |
| XGBoost | 2.1.4 |
| CatBoost | 1.2.10 |

The complete resolved environment is `requirements.lock`; installation is in ignored `automl/.venv/`.

## Implementation decisions and limits

- AutoGluon 1.5.0 is selected deliberately: the examined 1.6.3 Linux extra pulls `xgboost-cpu`, while FLAML pulls a different version of `xgboost`; both distributions install the same Python module. The chosen pair shares one XGBoost installation. Added `typing-extensions` explicitly because AutoGluon 1.5.0 imports it without a sufficient declared dependency on this Python version.
- AutoGluon's feature generator is fitted on training rows only, preventing its default joint train/validation feature-generation fit. Weighted ensembles are disabled because their validation-label weight fit exceeds the stated selection/early-stopping-only rule. Default medium-quality CPU model configurations otherwise remain available, excluding NN_TORCH/FASTAI. FLAML retains its default automatic estimator list.
- The leaderboard uses cached validation scores rather than predicting validation again with `leaderboard(val)`.
- Framework time budgets are search budgets, not hard process deadlines. Actual durations are recorded. There is no assumed study budget or automatic launch.
- Original byte hashes may not be reproducible with a different MLCA pandas/PyArrow environment. Such exports stop; they do not create replacement splits. Existing runs without any preserved membership file remain unavailable.
- The feature generator, preprocessing and genuine framework fitting have not been exercised end to end. Passing mocked contracts and import checks does not establish a successful training run.

## Deliberately not performed

Per the author's latest instruction: **neither local synthetic smoke tests nor the Codespace development-table smoke test was performed.** There are no smoke-test `result.json` excerpts or AutoML results to report. No MLCA command, model API call, real-table training, held-out/biomedical data read, cloud action, push or upload was performed. MLCA source and paper files were not changed.

Next gate: author/deployment-agent review, then separately authorized smoke tests. Biomedical task specs and final comparison time/CPU parameters remain scientific preregistration inputs.
