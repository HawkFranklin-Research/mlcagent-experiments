# Work order: planted-flaw test of MLCA's safeguards (R1-5)

Author decision, 24 October 2026. Reviewer 1, comment 5:
> "A small evaluation involving deliberately introduced known flaws—for example, overlap between training and test sets or preprocessing fitted with access to the test set—could demonstrate the behavior of the safeguards, including failures that remain undetected."

## What this is, and what it is not

- **What it is:** a test of the **frozen engine**. We write flawed analysis scripts ourselves and submit each one through MLCA's real step runner, exactly where a Worker's script would go. We record what happens:
  - **refused** by an enforced check;
  - **warned**: the step completed, but the engine logged a warning;
  - **caught only by the external scorer**;
  - **undetected**.
- **What it is not:** an agent run. There are no model calls, no DeepSeek, and no change to MLCA. Eval-02 stays the last agent run (stopping rule, 23 October). The model is MLCA's built-in scripted fake provider (`kind: fake`), which returns our scripts verbatim.
- **Nothing existing is modified:**
  - `mlca-revized` is read-only;
  - no file outside `safeguard-tests/` changes, except one line each in `RESULTS.md`, `PREREGISTRATION.md` and the `README.md` index.

## Version

The engine must be `freeze-v2` (`7dc0aed`, rails-0.15).
- Before running, check that `git -C ../mlca-revized diff --quiet freeze-v2 HEAD -- src templates rails` is clean, and record both commits. On 24 October, `HEAD` (`bfe38f0`) differs from the tag only in `docs/HANDOFF.md`.
- If the tree differs, use a detached checkout: `git -C ../mlca-revized worktree add --detach ../mlca-revized-freeze-v2 freeze-v2`. Put its `src` first on `sys.path`. Do not create a branch.
- Use the interpreter `../mlca-revized/.venv/bin/python`. Keep sandbox level A, the same as the studies (check that `mlca.sandbox.level.detect()` returns A).

## How to drive the engine (verified in MLCA's own tests)

- `tests/conftest.py::project_factory`:
  - writes a fake-provider script (YAML list of `{match: {role, action_id}, text}`);
  - builds the config `{"providers": {"offline": {"kind": "fake", "script": ..., "request_cost_bound_usd": 0}}, "roles": {"worker": {"provider": "offline", "model": "scripted"}}}`;
  - calls `mlca.project.init.initialise(folder, [data], template, answers, config)`.
- `tests/helpers.py`: `choose_action(session, action_id)` and `advance_to(session, event)` drive a `mlca.api.Session` event by event. They confirm the brief and the model selection, and they need an explicit `session.unlock_test("UNLOCK TEST")` before `test`.
- `tests/integration/test_test_lock.py` shows the test-lock refusal (`MlcaUserError`, "locked test"). `mlca.runner.run_step` applies `static_checks.check`; a refusal is categorised `fit_rule` or `static_check` (`run_step.py` around line 422).
- Template `tabular_classification` events: `onboard, data, eda, split, preprocess, features, model, validate, test, interpret, report`. Actions are in `templates/tabular_classification/actions/` (`freeze_stratified`, `freeze_group`, `fit_preprocessing`, `select_features`, `train_logistic`, `score_validation`, `score_test`, …). Use the regression template for Bike Sharing.
- External scorer: `scripts/score_leakage.py::score(project, expected)`. This is the same scorer as every study, with `expected` taken from `../mlca-revized/benchmarks/manifest.yaml` (leakage, group and time columns).

Copy what you need from `tests/helpers.py` into this folder. Do not import from MLCA's `tests/` at run time.

## The flaws

Every flaw has a **clean control**: the same step written correctly. The control must pass. A check that refuses everything proves nothing.

| ID | Flaw | Table | Planted at | Expected check (to be confirmed by the run, not assumed) |
|---|---|---|---|---|
| F1 | Scaler fitted on all rows before the split | synthetic (as in MLCA's tests) and Breast Cancer | `fit_preprocessing` | fit-on-training-rows rule / static check |
| F2 | Test rows copied into training | synthetic | `train_logistic` | test lock |
| F2b | Validation rows copied into training | synthetic | `train_logistic` | fit-row rule (the runtime records the fitted row indices) |
| F3 | Feature selection on all rows, before cross-validation | synthetic | `select_features` | fit-on-training-rows rule |
| F4a | Target-derived column used as input, declared in setup | Heart Disease (`num`) | `train_logistic` | `fit_record` refuses excluded columns |
| F4b | The same, not declared in setup | Heart Disease (`num`); Bike Sharing (`casual` + `registered`) | `train_logistic` | probably engine-undetected; external scorer |
| F5 | The same group in train and test (stratified instead of grouped split) | Anuran Calls (recording) | `freeze_stratified` instead of `freeze_group` | joint grouping rule, if the group column is declared |
| F6 | Time order broken (random split of a time series) | Bike Sharing | split | chronology rule, if the time column is declared |
| F7 | Reading the locked test set before `test` | synthetic | `score_validation` | test lock |
| F8 | Decision threshold tuned on the test set after unlocking | synthetic | `score_test` | likely only logged as test exposure; record what happens |
| F9 | Wrong metric (accuracy reported for a heavily imbalanced target) and swapped labels | synthetic | `score_validation` / `score_test` | expected undetected; the reviewer asked to show what slips through |
| F10 (optional) | Network access, or writing outside the project, from a script | synthetic | any | sandbox (R1-6b) |

Further flaws may be added, but **write every flaw into `flaws.yaml` before the first run**. Commit `flaws.yaml` and the scripts, then run. Do not change a flaw after seeing its outcome; a corrected flaw gets a new ID.

## Files to write (all in `safeguard-tests/`)

- `flaws.yaml`: one entry per flaw and per control. Fields: id, description, reviewer link, table, template, the declared setup (answers), the step, the planted script (verbatim), and the control script.
- `run_flaws.py`: for each entry, it does the following, then writes the outputs below.
  1. Build a fresh project under `runs/<date>/<id>/`, using a fake-provider script whose response for the planted action is the flawed script; every other action uses the clean fixture.
  2. Drive the session to the step.
  3. Record what happened:
     - the exception or refusal (type, category, message);
     - the step status;
     - warning events from the event log;
     - whether the run could continue to the report.
  4. Run the external leakage scorer when the project got far enough.
- `runs/<date>/matrix.csv` and `runs/<date>/MATRIX.md`: one row per flaw and control. Columns: flaw, table, outcome (`refused` / `warned` / `external_only` / `undetected` / `control_passed` / `control_failed`), the check that acted (engine rule or scorer criterion), the message (value-free), and the path to the run.
- `runs/<date>/environment.json`: both commits, rails version and SHA-256, Python, the package lock, sandbox level, host.
- `README.md`: purpose, how to run (`../mlca-revized/.venv/bin/python safeguard-tests/run_flaws.py --out safeguard-tests/runs/<date>`), the matrix, and limitations. Limitations to state:
  - the test checks the engine, not the agent's choices;
  - a declared-versus-undeclared setup decides several outcomes;
  - an undetected flaw is a finding, not a bug to fix now (no change after the freeze).
- `tests/test_safeguard_flaws.py` (in this repository's `tests/`): a fast check that `flaws.yaml` parses, every flaw has a control, and the matrix writer works on a stub result.

## Rules

- No model or network calls. The fake provider only.
- No edits to `mlca-revized`, `scripts/`, or any existing run folder.
- If a control fails, or a flaw makes the engine crash in a way that is not a refusal, record it as it is. Do not fix the engine. A crash in the engine is also a finding.
- Run once. The outcome is deterministic. Record the run date and keep all project folders (they are small).
- Commit locally on `master` (code, `flaws.yaml`, the run, the README); no push. Do not commit while an eval-02 resume or pooling commit is in progress in this repository; check `git status` first.

## Report back

1. The matrix: each flaw, its outcome and the check that acted.
2. Controls passed out of total.
3. Anything undetected.
4. Engine crashes.
5. The commit.
6. Time taken.

Then copy `MATRIX.md`, `matrix.csv` and `environment.json` to the paper side, `ES/submission2/experiments/results/safeguard-tests/`, with the source commit in a README. The paper uses this as the safeguards table (enforced / warned / external / undetected).
