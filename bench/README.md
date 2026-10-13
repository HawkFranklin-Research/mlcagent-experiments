# bench: `mlca bench` on standard public tables

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
