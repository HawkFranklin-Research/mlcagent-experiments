# bench: `mlca bench` on standard public tables

> **15 October 2026: this folder becomes the development set** (study design in `../README.md`; implementation in `../../mlca-revized/plan.md`, "Organized plan"). About 10 public tables, simple and complex, covering classification (binary, multiclass, multi-label), regression, clustering and representation learning, are run autonomously with open and closed models. Each failure is analysed and fixed as a rail, logged in a rails ledger with the failure that caused it. **2–3 held-out tables** are kept apart and run only after the freeze. The two survival tables leave development (survival is evaluated outside the rails). The table list and the folder name are still to be decided. The runs and notes already here (smoke runs, `validation/2026-10-15-nemotron/`) are development runs and stay as they are. The text below describes the earlier setup.

Measures MLCA v2 as an agent: completion, failures by category, automatic recovery, Navigator fallbacks, human and policy interventions, time split, model calls, tokens and cost. It uses built-in templates and choices only (no custom instructions), on the six tables in `../../mlca-revized/benchmarks/datasets/`.

| Path | What |
|---|---|
| `cards/` | Task cards (one per table and setting). `smoke_heart_gemma.yaml` is the first smoke card. |
| `models/` | Model catalogue snapshots and the first listing script (now built in as `mlca models`) |
| `runs/<task>/<model-slug>/` | `mlca bench` outputs; `runs/_smoke/` holds the first two smoke runs (2026-10-14, both stopped by provider rate limits) |

## Models (chosen 2026-10-13)

| Model ID | Access |
|---|---|
| `thinkingmachines/inkling-small:free` | free |
| `google/gemma-4-31b-it:free` | free |
| `nvidia/nemotron-3-super-120b-a12b:free` | free |
| `liquid/lfm-2.5-2.6b:free` | free |
| `openai/gpt-6-luna` | paid |
| `anthropic/claude-haiku-5.5` | paid |

- `openrouter/free` is a router (a different model per request) and is for local testing only.
- A model that disappears from OpenRouter before the runs is recorded as dropped.

## Notes

- Free OpenRouter models are paced to 16 requests per minute. Rate-limited steps wait and retry; waiting is reported as `provider_wait_s`.
- In `lung_cancer_survival.csv`, use `status` as the event column (`event` is all zeros).
- Before measured runs: freeze the MLCA version and commit `PREREGISTRATION.md` (models, cards, N, settings, approval policy, budgets, metrics, failure rules).
