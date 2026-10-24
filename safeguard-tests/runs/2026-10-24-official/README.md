# Official planted-flaw run, 24 October 2026

Answers Reviewer 1, comment 5 of Expert Systems 4881730: which deliberately introduced analysis flaws the frozen MLCA engine detects, which it only lets through to the external scorer, and which pass undetected.

- Protocol: [`../../WORK_ORDER.md`](../../WORK_ORDER.md). Flaws: [`../../flaws.yaml`](../../flaws.yaml) (fixed before any flaw was run).
- Engine: `freeze-v2` = `7dc0aedfb674e102f3cc96a92931a1a382eaaea0` (rails 0.15). `HEAD` (`bfe38f0`) differs from the tag only outside `src`, `templates` and `rails`; `git -C mlca-revized diff --quiet freeze-v2 HEAD -- src templates rails` exited 0 before the run.
- Harness: `caf28370` (`Safeguard harness: controls pass`); flaw catalogue at `51476857`.
- Model: MLCA's scripted fake provider (`kind: fake`), worker role only. No navigator, no model calls, no network attempts (`environment.json`), sandbox level A. Each flawed script was written by hand and returned verbatim where the Worker's script would go.

## Result

26 rows: 13 flaws and their 13 clean controls. All 13 controls reached the report; no engine crash occurred. Full table: [`MATRIX.md`](MATRIX.md) and [`matrix.csv`](matrix.csv); per-case detail in `runs/<case>/result.json` and each project's `.mlca/log/events.jsonl`.

| Outcome | Cases |
|---|---|
| refused by an enforced check (9) | F1-synthetic, F1-breast-cancer, F2, F2b, F3, F4a, F5, F6, F7 |
| warned (0) | none |
| external scorer only (2) | F4b-heart, F4b-bike |
| undetected (2) | F8, F9 |

### What acted, and how

- **F1 (both tables), F2b, F3 — fit on non-training rows.** `mlca_runtime.fit_record` compares the row index of every fitted object with the frozen training index. The planted scripts concatenate validation (or validation plus test-request) rows, so the fit is refused: "Fit rows must match the frozen training index" (144 expected, 192 received; 341 vs 454 for Breast Cancer).
- **F2, F7 — reading the locked test partition.** The test partition is not among the inputs of a step that does not read it (`reads_test: false`). At the model and validation events the input dictionary has no `test` key and no test unlock had occurred; the planted `m.inputs()['test']` fails with `InputKeyError`. The test lock acts by withholding the input, not by rejecting a request.
- **F4a — declared target-derived column as input.** `num` is declared excluded in setup; `fit_record` refuses "Target, excluded or group columns cannot be fitted as inputs: ['num']".
- **F4b-heart, F4b-bike — undeclared target-derived column.** Nothing in the engine objects: the model fits on `num` (Heart Disease) and on `casual` + `registered` (Bike Sharing), the workflow reaches the report. Only the external leakage scorer flags them (`fitted_columns: false`, violations `['num']` and `['casual', 'registered']`).
- **F5 — same recording group across partitions.** The harness requested `method=stratified` although `group_column=RecordID` was declared. The engine did not honour the request: `freeze` substituted the declared method, the frozen split records `method=group`, and the scorer confirms group-disjoint partitions. The flaw never reaches the data. (Recorded as `refused`: the flawed request was overridden, not executed; the engine raises no error.)
- **F6 — broken time order.** The harness requested `method=random` although `dteday` and `hr` were declared as time columns. `freeze` substituted `method=time`; the frozen split is chronological (`time_ordered: true`). Same override mechanism as F5.
- **F8 — threshold tuned on the unlocked test set.** After the explicit unlock, the planted script chose the threshold on the test labels (0.65 of the grid). The step completed, recorded its metric and row outputs, and no warning was logged. Undetected.
- **F9 — wrong metric and inverted labels.** The planted script reports accuracy, with inverted predictions, under the name `roc_auc` on an imbalanced target (0.149). The metric is recorded as declared; no check compares the value with the metric's definition. Undetected.

## Limitations

- This measures the **engine's enforced checks** and the external scorer, not the agent's choices. The scripts are ours; MLCA's fake provider returns them verbatim.
- Several outcomes depend on the declared setup: F4a (declared exclusion) is refused, F4b (same column undeclared) is invisible to the engine. That contrast is the result, not an inconsistency.
- The engine records F5/F6 as an override of the requested split parameters; the matrix label `refused` means the flawed request was not executed.
- Undetected flaws are findings. Nothing in MLCA was changed after the freeze; F8 and F9 remain undetected in this version.
- One run per case; the engine path is deterministic (fixed seeds, fixed splits). Data copies, split manifests and per-project environments are gitignored (`**/.mlca/data/`, `**/.mlca/env/`); the event log, scripts, outputs, `result.json` and the split parameters/counts in `SplitFrozen` events are kept.
- Synthetic tables are small; the flawed scripts are minimal and target one safeguard each.
