# Instructions for AI agents working in this repository

This repository holds MLCA v2 (`mlcagent`) evaluation runs: task cards, study protocols, data-preparation scripts, the code MLCA's models generated, logs, token and cost records, metrics and figures. MLCA's code is in `../mlca-revized/` (`HawkFranklin-Research/mlcagent`).

## Study design (15 October 2026): read first

The design is in `README.md` ("Study design"). Rules that follow from it:
- **`bench/` is the development set** (about 10 public tables) where the agent's rails are built, plus **2–3 held-out tables** that are not run, smoke-tested or opened by the agent before the freeze.
- **Rails ledger:** every rail added during development records the failure that caused it (dataset, model, run path, step, failure category) and the commit that added it.
- **Freeze before held-out and evaluation runs:** version, rails and prompts tagged; `PREREGISTRATION.md` committed. No change to the agent after the freeze; a later fix is a later version, and its runs are labelled as such.
- **`bio-bench/` is the evaluation** (Phase 2): the frozen agent receives each dataset with a short starting brief, written in advance and kept verbatim with its word count. Arm A autonomous; arm B human-on-the-loop (halts and overrides recorded).
- Survival analysis is not part of development; it appears only in evaluation.
- Unfinished parts of the earlier plans are stopped. The step-by-step `STUDY.md` protocols are kept for their data preparation and aims.

## Two benchmarks, kept apart (earlier description)

- **`bench/`** runs `mlca bench`: standard public tables (`../mlca-revized/benchmarks/datasets/`), built-in templates and choices only, no custom instructions. It measures MLCA as an agent.
- **`bio-bench/`** runs the biomedical studies S1–S4: built-in choices first, then custom instructions. It measures both the science and the agent.

## Bio-bench: ambition and what counts as success

- The goal is ambitious: reproduce every step of each study protocol, including the most complex (S4), with MLCA.
- **If a step cannot be achieved reliably with MLCA, that is an acceptable outcome.** Record it as *not achieved*, with the reason (engine limitation, scope refusal, model failure, provider failure). Then continue or stop as the protocol says.
- Never force a step with hand-written code passed off as MLCA's, never fabricate or back-fill a result, and never drop a failed run.
- Missing capability is a finding, not a defect to hide.

## Custom instructions are mandatory research artefacts

Every custom instruction is part of the published method. For each one issued, keep with the run:
- the exact text;
- the event and step it was issued in;
- the model;
- the scope decision (accepted, redirected or refused, and the reason);
- the resulting step and its outcome;
- the word count.

MLCA's event log and `.mlca/llm/` already record the text and the decisions. Export them to `custom_instructions.md` in each run's curated folder. Do not paraphrase them, and do not edit them after the run.

## What may and may not be pushed

- **Push:** protocols (`STUDY.md`), task cards, data-preparation scripts, generated step scripts, event logs, prompts and responses (`.mlca/llm/`), usage and token records, metrics, reports and figures.
- **Never push** (enforced by `.gitignore`):
  - `*.local.md` / `*.local.yaml`: private planning notes and local paths;
  - `.mlca/private/` (sealed test partitions, keys), `.mlca/env/`, `.mlca/data/`, `data_prep/`.
- Pushable files describe the studies as they are run here; private context belongs in `*.local.md`.
- No push, remote or upload without the author's explicit go-ahead.
- **Planned homes:** code and small artefacts → GitHub `HawkFranklin-Research/mlcagent-experiments`; prepared study tables and full run archives → Hugging Face dataset `HawkFranklin-Research/mlcagent-bio-bench` (name to confirm). Both mirror this layout.

## Conventions

- **Model slug:** the model ID with `/` → `__` and `:` → `_`.
- **Run layout:** `<bench|bio-bench>/<study-or-task>/runs/<model-slug>/<bench-output>/run-NNN/`.
- **Paths:** local paths come from `paths.local.yaml` (copy `paths.local.yaml.example`).
- **Rules:** commit a `PREREGISTRATION.md` before measured runs; freeze the MLCA version first; record the `mlcagent` commit in each run.
