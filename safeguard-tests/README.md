# Safeguard tests: deliberately flawed analyses against the frozen engine

Answers Reviewer 1, comment 5 (Expert Systems 4881730): which problems MLCA detects automatically, which only produce warnings, which depend on the researcher, and which go undetected.

Flawed analysis scripts, each paired with a correct control, are submitted through MLCA's real step runner at `freeze-v2`. MLCA's scripted fake provider returns them verbatim. The test therefore makes **no model calls and no agent runs**; it measures the engine's enforced checks and the external leakage scorer, not the agent's choices. Author decision 24 October 2026; protocol in [`WORK_ORDER.md`](WORK_ORDER.md).

Status: **done** (24 October 2026). Official run: [`runs/2026-10-24-official/`](runs/2026-10-24-official/README.md), commit `199fb5df`.
- 13 flaws and 13 controls; all 13 controls passed.
- 9 flaws refused by enforced checks, 2 caught only by the external scorer (F4b), 2 undetected (F8, F9), 0 warned.
- 0 model calls, 0 network attempts, no engine crash.
- The first attempt (`runs/2026-10-24/`) failed in the harness; it is recorded below, not a result.

## Record: harness failure on 24 October 2026 (`runs/2026-10-24/`, partial, not a result)

The first attempt (commits `67fbf58d`, `9d8dacb2`) was stopped after cases F1 and F2. Its outcomes are **harness errors, not safeguard results**:
- The runner kept only actions with role `worker`; template actions are labelled `primary` or `supporting`. The fake provider therefore had no scripts (`No fake response for worker/write/0`).
- The runner did not isolate MLCA's user configuration. The real `~/.config` defines a navigator on `openrouter/free`, so in F2 the navigator made 3 requests to free OpenRouter models. The requests carried the synthetic table's summary (about 8.6 kB: facts, inventory summary, template text), cost $0, and contained no study or biomedical data. This broke the work order's no-network rule.
- The control variant ran a generic script instead of the case's `control_script`, and a format-string bug in the model-step script was hidden by the first bug.

Fixed before any further run (harness only; `flaws.yaml` unchanged):
- an empty `XDG_CONFIG_HOME`/`XDG_DATA_HOME` per run, as in MLCA's own test fixture, so there is no navigator role and MLCA uses its deterministic fallback options;
- provider keys removed from the environment;
- every non-Unix socket connection refused in-process, and the run stops if one is attempted;
- all template actions scripted;
- the case's own control script used;
- `--cases` and `--variants` options for harness checks.

The folder `runs/2026-10-24/` is kept as this record. The official run uses a new folder.

## Amendment before the official run (24 October 2026)

During Phase A (controls only, no flaw variant run), the Heart Disease controls failed: the table has missing values (`ca` ×4, `thal` ×2), and the F4a and F4b-heart scripts fitted `LogisticRegression` on raw columns. The planted versions would have crashed the same way before any safeguard could act. All four scripts now impute (median, inside the fitted pipeline). The flaw is unchanged: the planted scripts use `num`, and the controls drop it. No other test table has missing values. Recorded at the top of `flaws.yaml`.

## Result: official run, 24 October 2026 (`runs/2026-10-24-official/`)

All 13 controls reached the report; 13 flawed scripts produced 9 refusals, 2 external-only detections, 0 warnings and 2 undetected flaws. No engine crash; no model calls; no network attempts. Full table and per-case evidence: [`runs/2026-10-24-official/README.md`](runs/2026-10-24-official/README.md), [`MATRIX.md`](runs/2026-10-24-official/MATRIX.md), [`matrix.csv`](runs/2026-10-24-official/matrix.csv).

| Outcome | Cases |
|---|---|
| refused | F1-synthetic, F1-breast-cancer, F2, F2b, F3, F4a, F5, F6, F7 |
| warned | none |
| external only | F4b-heart, F4b-bike |
| undetected | F8, F9 |

Limitations: the test measures the frozen engine, not the agent's choices; declared versus undeclared setup decides several outcomes (F4a vs F4b); F5/F6 are recorded as `refused` because the engine overrode the requested split method rather than raising an error; an undetected flaw is a finding, and nothing was changed after the freeze.

