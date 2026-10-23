# MLCA v2 results: what was run, in what order, and where it is

Status on 22 October 2026. This file describes every set of agent runs in this repository. It is written to serve later as the dataset card for the raw-results upload. Main-study raw results are backed up privately at `HawkFranklin-Research/mlcagent-runs`, verified revision `d39aee54ed033c1ace8128aefec5a3e86f5f5255`. See [the offload and restore record](docs/storage/huggingface-offload-2026-10-22/README.md).

## 1. The study in one paragraph

MLCA v2 is an autonomous agent: it receives a data table and a short brief, decides what the study is, writes and runs the analysis code step by step, and writes a report. Its behaviour is guided by **rails**: fixed, generic instructions and checks that are part of the agent. The rails were built the way a model is trained:
- **developed** on public tables;
- **frozen**;
- **tested** on tables they never saw (held-out tables);
- **evaluated** on the biomedical studies.

The main question is not model accuracy, but: how often does the agent finish, where does it fail, how many repairs does it need, and what does it cost? Protocol: `PREREGISTRATION.md`.

## 2. Terms

| Term | Meaning |
|---|---|
| **Study** (or run) | One fresh agent run on one table, from the data to the report. Each has its own folder. |
| **Batch** | A group of studies run at the same time on one unchanged version of the agent. |
| **Success** | Three conditions, scored outside the agent: (1) the workflow completed; (2) the agent's setup (problem type, target, group and time columns) agrees with the researcher's declared task; (3) the leakage checks hold (no target-derived or identifier column used, no group split across partitions, time order respected). |
| **Rails version** | `rails-0.3` … `rails-0.14`. Every rail is logged with the failure that caused it (`../mlca-revized/rails/LEDGER.md`). |
| **Acceptance** | The test that decides the development is finished: each table runs 3 times on one unchanged version and must succeed in at least 2 of the 3. |
| **Freeze** | A git tag on the accepted agent (`freeze-v1` = `a9f4e35`). Nothing changes after it. |

## 3. The main study sets

These are the sets the paper reports. Model throughout: DeepSeek V4.1 Flash, temperature 0. Up to batch-08 it was called through OpenRouter (`deepseek/deepseek-v4.1-flash`); from acceptance-01 on, through the DeepSeek API directly (`deepseek-flash`, reasoning effort low).

### Version 1: development, acceptance, freeze

| # | Set | Folder | Agent version | Studies | Successful | What it is |
|---|---|---|---|---:|---:|---|
| 1 | Development batches 01–08 | `bench/dev-loop/batch-01` … `batch-08` | rails-0.3 → 0.9 | 45 | 12 | Rails built from failures. Batch-01 ran all 12 tables; each later batch reran the failed tables after one round of fixes. |
| 2 | Acceptance-01 | `bench/comparison/acceptance` | rails-0.9 | 36 (12 × 3) | 22 | First acceptance attempt. |
| 3 | Development batch-09 | `bench/dev-loop/batch-09` | rails-0.10 | 7 | 5 | Fixes for acceptance-01 failures. |
| 4 | Acceptance-02 | `bench/comparison/acceptance-02` | rails-0.11 | 36 (12 × 3) | 24 | Second full acceptance attempt (see §4). |
| 5 | Pooled acceptance batches 10–13 | `bench/dev-loop/batch-10` … `batch-13` | rails-0.12 → 0.13 | 16 | 12 | Tables still short of 2-of-3, rerun on the corrected version; batch-13 is two confirmation runs on rails-0.13. |
| — | **Freeze `freeze-v1`** | `../mlca-revized` tag | `a9f4e35`, rails-0.13 | — | — | |

**Which version each table was accepted on** (pooled rule; the frozen version is the last one):

| Table | Accepted on | Evidence |
|---|---|---|
| Breast Cancer Diagnostic, Adult, Dry Bean, Anuran Calls, Abalone, Energy Efficiency, Seeds, Online Retail, Optical Digits | rails-0.11 | acceptance-02 (≥2 of 3) |
| Heart Disease, Covertype | rails-0.12 | batches 10–11 |
| Bike Sharing | rails-0.13 | batch-12 |

### Version 1: tested after the freeze

| # | Set | Folder | Studies | Successful | What it is |
|---|---|---|---:|---:|---|
| 6 | Held-out tables | `bench/heldout/heldout-01` | 12 (4 × 3) | 5 | Phase 1b. Bank Marketing 2/3, Wholesale Customers 3/3. Steel Plates 0/3 and Wine Quality 0/3, all because of setup disagreements. |
| 7 | AutoML comparison | `bench/automl`, `bench/automl-heldout` | — | — | AutoGluon 1.5.0 and FLAML 2.7.0 on the same splits as MLCA (7 development and 3 held-out tables). Not agent runs. |
| 8 | Biomedical evaluation, arm A | `bio-bench/runs/deepseek-flash/eval-01` | 27 (9 × 3) | 1 | Phase 2. Failure causes and one scoring deviation in its `EVALUATION_SUMMARY.md`. Triggered the reopening rule. |

### Version 2: the reopening (rule 7), in progress

| # | Set | Folder | Agent version | Studies | Successful | What it is |
|---|---|---|---|---:|---:|---|
| 9 | Development v2, batch-01 | `bench/dev-loop-v2/batch-01` | rails-0.14 | 15 | 10 | 12 tables plus 3 wide gene-expression tables (GLI-85, SMK-CAN-187, GCM). |
| 10 | Development v2, batch-02 | `bench/dev-loop-v2/batch-02` | rails-0.15 | 9 (3 × 3) | 9 | Acceptance of the 3 wide tables, 3/3 each. The 12 v1 tables keep their acceptance; batch-01 is their regression check. |
| — | **Freeze `freeze-v2`** | `../mlca-revized` tag | `7dc0aed`, rails-0.15 | — | — | |
| — | eval-02 | `bio-bench/runs/deepseek-flash/eval-02` | freeze-v2 | — | — | To come. Repeats eval-01 exactly; the last agent run of this revision. |

**Total in the main study so far:** 203 agent studies (sets 1–6, 8–10).

## 4. Why there are two full acceptance batches (36 + 36)

- **Acceptance-01** (17 October, rails-0.9) ran all 12 tables 3 times. The rule then in force (work order 05) said: if any run fails, fix the cause and rerun the whole acceptance.
  - It was short on four tables: Covertype 0/3, Anuran Calls 0/3, Online Retail 0/3, Bike Sharing 1/3.
  - Eight tables had 2 or 3 successes.
- After batch-09's fixes, **acceptance-02** (18 October, rails-0.11) therefore ran all 12 tables 3 times again, including the eight that had already passed.
- On 19 October the author replaced that rule with **pooled acceptance**: a table that reaches 2-of-3 keeps its result, and only tables that fell short run again (3 times). From then on no full rerun was made (batches 10–13).

Under the pooled rule, acceptance-02's rerun of the eight passing tables would not have been needed. Both batches are kept and reported, as every run is (study design, rule 5). Acceptance-01 is the first acceptance attempt; the per-table acceptance in §3 follows the pooled rule from acceptance-02 on, as recorded in the rails ledger.

## 5. Not part of the main study (archive)

These were software checks and debugging before the measured series began. They are kept for completeness, are not reported as results, and are not compared with the main sets.

| Folder | What | Why excluded |
|---|---|---|
| `bench/dev-loop/round-01`, `round-02` | First development rounds, `openrouter/free` (the model changed between requests), tables run one after another | Superseded by the batch series on 16 October; the study design keeps them as evidence but never compares them. |
| `bench/validation/2026-10-15-nemotron` | Debugging batch with a free Nemotron model | Software check before the design pass. |
| `bench/runs/_smoke`, `bench/runs/heart_disease_cleveland_uci`, `bench/SMOKE_*.md` | Smoke runs (driver and OpenCode/MCP experiments) | Smoke tests, not measurements. |
| `bio-bench/*/runs/reference/`, `bio-bench/acceptance/` | Reference analyses with no model calls | Not agent runs; never uploaded before acceptance of the paper. |

## 6. What one study folder contains

```
<set>/studies/<table>-<replicate>/attempt-001/
  worker-result.json         outcome, failure category, tokens, time
  setup-agreement.json       external setup score (survival rescoring: setup-agreement-rescored.json)
  leakage-score.json         external leakage checks
  setup-001/.mlca/           the setup request and response (what the agent inferred)
  run-001/                   the MLCA project:
    scripts/                 every analysis script the agent wrote
    tables/ reports/         its outputs and report
    PROJECT_LOG.md           step-by-step log
    .mlca/log/               hash-chained event log
    .mlca/llm/ requests/     model requests and responses
    .mlca/usage.jsonl        tokens and timing per request
    .mlca/egress.jsonl       what was sent to the model provider
    .mlca/steps/             each step's attempts, scripts, stdout/stderr
    .mlca/data/ env/         copies of the input table and Python environment (regenerable; not uploaded)
<set>/batch.json, REPORT.md, cost-ledger.json, resources.jsonl
```

Each set's `REPORT.md` has one row per study: completed, setup agreement, leakage checks, furthest step, failure category, repairs, requests, tokens and wall time.

## 7. Inputs and versions

| What | Where |
|---|---|
| Agent code, rails, ledger | `../mlca-revized` (`freeze-v1` = `a9f4e35`; v2 in progress on master) |
| Development and held-out tables | `../mlca-revized/benchmarks/manifest.yaml` (source, licence, SHA-256 for each) |
| Biomedical tables | Hugging Face `HawkFranklin-Research/mlcagent-bio-bench`, revision `111de16`, fetched by `scripts/fetch_bio_bench.py` |
| Biomedical tasks, briefs, decisions | `bio-bench/manifest.yaml`, `bio-bench/briefs/`, `PREREGISTRATION.md` |
| Scorers | `scripts/score_inference.py`, `scripts/score_leakage.py`, `scripts/score_unsupervised.py` |

## 8. Storage

- **Git** holds the curated evidence: reports, batch records, cost ledgers, summaries, and the per-study records listed above, minus row-level Parquet files, split-membership CSVs, control files and data copies.
- **Raw results** are backed up privately with per-file and per-archive SHA-256 manifests. After remote verification, copied data/environments and untracked result copies were removed locally; curated Git-tracked evidence remains. The earlier 20 GB estimate for copied inputs/environments was incorrect: approximately 15.25 GiB was private row-fingerprint storage, excluded from the upload; the author had it deleted on 23 October 2026 (about 17.9 GiB with the validation folders). See the offload record above for exact cleanup, receipts and restoration.
