# mlcagent-experiments

Evaluation runs of **MLCA v2** (`mlcagent`, code in `../mlca-revized/`, future `HawkFranklin-Research/mlcagent`). It keeps everything a run produces that is worth sharing: task cards, study protocols, data-preparation scripts, the step scripts written by each language model, prompts and responses, event logs, token and cost records, metrics, reports and figures.

Planned remote: `HawkFranklin-Research/mlcagent-experiments`. Local only for now; nothing is pushed. **AI agents: read `AGENTS.md` first.**

| Folder | What | Inputs |
|---|---|---|
| `bench/` | `mlca bench`: agent metrics on six standard public tables, built-in workflow only | `../mlca-revized/benchmarks/datasets/` |
| `bio-bench/` | Biomedical studies S1–S4 (anemia CBC; CPTAC-BRCA receptors; BRCA PAM50 transfer TCGA→CPTAC; HNSCC subtype discovery and prognostic signature). Built-in choices, then custom instructions | public cohorts prepared by each study's `prepare_data.py` into `data_prep/` (git-ignored) |
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
