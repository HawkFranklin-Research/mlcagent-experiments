# Safeguard tests: deliberately flawed analyses against the frozen engine

Answers Reviewer 1, comment 5 (Expert Systems 4881730): which problems MLCA detects automatically, which only produce warnings, which depend on the researcher, and which go undetected.

Flawed analysis scripts, each paired with a correct control, are submitted through MLCA's real step runner at `freeze-v2`. MLCA's scripted fake provider returns them verbatim. The test therefore makes **no model calls and no agent runs**; it measures the engine's enforced checks and the external leakage scorer, not the agent's choices. Author decision 24 October 2026; protocol in [`WORK_ORDER.md`](WORK_ORDER.md).

Status: **planned**; nothing has run yet. Files to come:
- `flaws.yaml`, committed before the first run;
- `run_flaws.py`;
- `runs/<date>/` (`matrix.csv`, `MATRIX.md`, `environment.json`, one project folder per flaw).
