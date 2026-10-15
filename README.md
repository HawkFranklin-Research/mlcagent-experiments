# mlcagent-experiments

Evaluation runs of **MLCA v2** (`mlcagent`, code in `../mlca-revized/`, future `HawkFranklin-Research/mlcagent`). It keeps everything a run produces that is worth sharing: task cards, study protocols, data-preparation scripts, the step scripts written by each language model, prompts and responses, event logs, token and cost records, metrics, reports and figures.

Planned remote: `HawkFranklin-Research/mlcagent-experiments`. Local only for now; nothing is pushed. **AI agents: read `AGENTS.md` first.**

## Study design (15 October 2026)

The question: **how often does an autonomous ML agent complete a study, where does it fail, how many repairs does it need, what does it cost, and how much human help does it need?**

| Phase | Where | Datasets | Agent may change? |
|---|---|---|---|
| 1 · Development: build the agent's **rails** (fixed instructions, prompts, engine checks) from observed failures | `bench/` | about 10 public tables, simple and complex: classification (binary, multiclass, multi-label), regression, clustering, representation learning | yes; every rail is logged with the failure that caused it (rails ledger) |
| Freeze | — | version, rails and prompts tagged; `PREREGISTRATION.md` committed | — |
| 1b · Held-out tables | `bench/` (held-out) | 2–3 public tables never run before the freeze | no |
| 2 · Evaluation | `bio-bench/` | biomedical studies S1–S4, each with a short starting brief written in advance | no |

Phase 2 runs in two arms: **A, autonomous** (no human after the brief) and **B, human-on-the-loop** (a human may halt any decision within a short window; otherwise the agent decides). Each arm uses open models (with tool calling) and cheaper closed models. Survival analysis is outside the rails: it appears only in evaluation. A fix found after the freeze is a later version and is labelled as such. Implementation: `../mlca-revized/plan.md`, "Organized plan (15 October 2026)". Unfinished parts of the earlier plans are stopped; their files are kept.

| Folder | What | Inputs |
|---|---|---|
| `bench/` | Development and held-out tables (Phase 1 and 1b; name may change). Earlier: `mlca bench` agent metrics on six standard public tables, built-in workflow only | `../mlca-revized/benchmarks/datasets/` |
| `bio-bench/` | Biomedical evaluation (Phase 2), studies S1–S4 (anemia CBC; CPTAC-BRCA receptors; BRCA PAM50 transfer TCGA→CPTAC; HNSCC subtype discovery and prognostic signature). Data preparation per study; starting briefs to be written | public cohorts prepared by each study's `prepare_data.py` into `data_prep/` (git-ignored) |
| `data_prep/` | Prepared study tables (large, git-ignored, regenerated) | — |
| `paths.local.yaml` | Local paths (git-ignored); template `paths.local.yaml.example` | — |

## How a run is stored

```
<bench|bio-bench>/<task-or-study>/runs/<model-slug>/<bench-output>/
  bench.json, bench_summary.json        run manifest and cross-run summary (mlca bench)
  openrouter_models_<date>.csv          model catalogue snapshot used by the run
  run-NNN/                              one MLCA project per run
    scripts/  reports/  PROJECT_LOG.md  generated step scripts and reports
    .mlca/log/  .mlca/steps/  .mlca/llm/  .mlca/requests/  usage.jsonl  egress.jsonl
    (.mlca/env, .mlca/data, .mlca/private are never pushed)
```

**Model slug:** the model ID with `/` → `__` and `:` → `_`, e.g. `google__gemma-4-31b-it_free`. Per-study cross-model summaries go in `<study>/summary/`.

## Reproduce

```bash
uv tool install --force --reinstall ../mlca-revized        # or: uv tool install mlcagent
mlca models --select openai/gpt-6-luna --select anthropic/claude-haiku-5.5
mlca bench run bench/cards/<card>.yaml --runs N --out bench/<task>/runs/<model-slug>/<date>
mlca bench summarize bench/<task>/runs/<model-slug>/<date>
```
