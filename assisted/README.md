# Assisted HNSCC runs: the released agent with pre-written expert instructions

> **Status: prepared, not run in this revision** (author decision, 24 October 2026). The revised paper evaluates the agent working alone, with no human instructions after the starting brief. The HNSCC result stays at 0/3 autonomous. This stage is kept ready for a later revision, if reviewers ask. Only the mechanics were checked: one smoke run on GBSG2, a public survival table outside the rails, in a scratch folder (not a result). No HNSCC run was made with these instructions.
>
> **What the smoke showed (24 October 2026):**
> - the scripted halt-and-instruct path works;
> - stage instructions are given and executed;
> - a failed step is paused and the repair instruction given;
> - the cap stops the run.
>
> Two issues remain to solve before any real use:
> 1. **The clustering instruction is refused at the features stage.** MLCA's scope classifier (the Navigator) refused it as belonging to a "clustering" stage, which the survival workflow does not have. The submitted HNSCC design (clustering, then classification, then survival validation) spans several workflow types, while MLCA runs one type per study.
> 2. **The repair wording is refused.** At validation, the repair instruction was refused three times as belonging to the model stage, because it mentions saving fitted objects. Repair text must be stage-neutral.
>
> `scope_check.py` (asks the scope classifier only, with no analysis) is ready for testing new wording; it has not been run.

**Question.** The released agent (rails 0.15) failed all three autonomous HNSCC survival studies in eval-02 (S4; survival is a task type kept outside the rails). How far does it get when a fixed set of expert instructions, written in advance, is given at fixed stages? The instructions follow the design of the submitted HNSCC study.

**Design** (author-approved, 24 October 2026; exploratory, reported apart from the 22/27 autonomous result):
- **Runs:** 3 fresh runs, identical to the eval-02 HNSCC studies in everything else: data, brief, external CPTAC cohort, model and settings, rails, sandbox, scoring.
- **Instructions** (`instructions-hnscc.yaml`), given through MLCA's own human-on-the-loop path:
  - at the first decision of a listed stage, the scripted human halts the agent's proposed choice once and gives the instruction, which MLCA runs as the user's custom step;
  - after a failed step, the repair instruction is given instead of the automatic retry;
  - at most 5 instructions per run.
- **No change to MLCA.** `run_assisted.py` replaces the autonomous drive loop only inside its own process.
- **Instruction content:** the method choices of the submitted study (`mlca_experiments/jun16_copilot_agent_results_case_study1.py`), with none of its results. Three deliberate differences are listed in the YAML header:
  1. clustering is fitted on training rows only;
  2. CPTAC is never used for selection;
  3. the classifier is fitted on all training rows.

  The instructions were drafted by the AI assistant from that code and approved by the author.
- **Not a clean test of human help:**
  - the instructions were written after seeing the autonomous failures;
  - the author knows the earlier analysis;
  - a fresh run can also differ by chance.

  Report: "with N pre-written instructions, X of 3 runs succeeded". Report it alongside the 0/3 autonomous result, never instead of it.

**Files**
- `run_assisted.py`: the runner.
- `instructions-hnscc.yaml`: the instructions, fixed before the first HNSCC run.
- `instructions-smoke-gbsg2.yaml`: a mechanics check on a public survival table outside the rails (GBSG2), run into a scratch folder. Not a result.
- `../tests/test_assisted.py`: logic tests with a stub session.

**Run** (laptop, from the repository root):
```bash
scripts/with_deepseek_key.sh ../mlca-revized/.venv/bin/python assisted/run_assisted.py \
  --instructions assisted/instructions-hnscc.yaml --out bio-bench/runs/deepseek-flash/assisted-hnscc
```

**Outputs**, per study `studies/s4_hnscc-assisted-00N/attempt-001/`:
- the MLCA project;
- `worker-result.json`, which includes the instructions used, their stage, trigger and word count;
- the controller's own scoring: `setup-agreement.json` and `leakage-score.json`;
- `assisted-result.json`.

`assisted-summary.json` sits at the top of the output folder. Every instruction is also in each project's event log (`DecisionWindowHalted`, `CustomInstructionSubmitted`).
