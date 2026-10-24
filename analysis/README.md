# Analyses of existing results (24 October 2026)

Local analyses of records that already exist. **No model calls and no new agent runs.** Every number comes from the committed run folders of eval-02 (the released version, rails 0.15), the biomedical AutoML results (`bench/automl-bio`) and the prepared S1 table. Each subfolder has its script, so every table can be regenerated with `../mlca-revized/.venv/bin/python analysis/<folder>/<script>.py`.

## 1. eval-02 records: cost, time, decisions, what was sent (`eval02/`)

Script `collect.py` → `studies.csv` (the 27 official studies), `decisions.csv`, `attempts_usage.csv` (every attempt, including infrastructure retries and records), `egress_summary.csv`, `cost.json`.

**Cost (R1-4, R2-2).** Each request is priced at DeepSeek's published rate for `deepseek-flash` (`DeepSeek-V4.1-Flash-1017`; api-docs.deepseek.com, read 24 October 2026). Per 1 M tokens at peak: cache hit $0.006, cache miss $0.30, output $1.20, with off-peak at half; peak and off-peak come from each request's timestamp. Output includes reasoning tokens.
- **All 27 studies: $2.08.** Including every infrastructure retry and record: **$2.38**.
- **Median per successful study: $0.067** (range about $0.05–0.19).
- 976 requests. Tokens: 9.47 M input (7.07 M served from the provider's cache), 2.37 M output, of which 1.97 M reasoning.

**Time per successful study** (medians):
- 38.5 requests;
- model time (sum of request latencies) 530 s;
- time running the agent's scripts 721 s;
- wall time 2,708 s.

The rest of the wall time is environment creation, data review, waits and setup. Wall time depends on the machine: the laptop for S1, S2 RNA ER/PR and proteome ER run 3; 4-vCPU cloud VMs for the rest, with the host recorded per study. By task, the median wall time ranges from 8 min (S1) to 69 min (S2 RNA HER2, VM). There was no human time in any run (arm A).

**Decision consistency across runs (R1-2a).** All three runs of each task agreed on:
- problem type and target;
- split method (stratified, seed 42, 60/20/20);
- model families (logistic regression and histogram gradient boosting; S4: Cox proportional hazards with a random split);
- the number of features fitted.

Recorded test metrics were identical across runs in most tasks. The **one divergence:** in S1 run 2 the agent excluded `Hemoglobin` as a suspected leak at setup (4 features against 5), and its test ROC AUC was 0.54 against 1.00 in runs 1 and 3. Duplicate removal was applied in all three S1 runs (next section). For the S2 and S3 tables, the data review excluded many sparse columns, identically in every run.

**What was sent to the model provider (R1-6b).** The `egress.jsonl` categories over all 27 studies are only:
- `inventory_summary` (the per-column profile);
- `setup_values`;
- `user_text` (the brief);
- `facts`;
- `template_spec`;
- `script` (the agent's own code, when repairing);
- `traceback` (error messages).

No category of individual records was sent, because sample rows are off by default. The largest single request was the setup profile: about 18 KB for S1, 0.59 MB for the proteome tables, 1.06–1.10 MB for the RNA tables, 0.18 MB for PAM50 and 28 KB for S4. Total sent per study ranged from 0.33 MB (S4) to 4.8 MB (S2 RNA PR run 2).

## 2. S1 anemia table (`anemia/`, for R2-5 and R1-7d)

Script `anemia_checks.py` → `anemia_checks.json`.
- **Duplicates:** 1,421 rows, only **534 unique** (887 exact copies).
  - In every eval-02 run, **MLCA's engine removed them before the split.** Its data review asks "Some rows are exact copies. Remove duplicates before splitting?" with removal recommended, and the autonomous policy accepted the recommendation. It is recorded as a `DataDecision` (`drop_duplicates`).
  - A random 80/20 split of all 1,421 rows, as in the submitted analysis, puts an exact copy of **90%** of the test rows into training (mean over 200 splits, range 85–95%).
- **The label is a haemoglobin rule:** `Result = 1` exactly when Hemoglobin < 12.0 for `Gender` 0 and < 13.5 for `Gender` 1. The rule reproduces the label for **100%** of all 1,421 rows (and of the 534 unique rows). The label is therefore defined by one of the inputs, so near-perfect accuracy is expected whenever Hemoglobin is used.
- The original source of the file is not recorded in the table. The author is confirming it.

## 3. Replay of stored code (`replay/`, for R1-2b)

Script `replay_check.py` → `replay_results.json`.
- **Method:** each selected study is copied to a scratch folder (replay opens the project for writing, so originals are never used). Then `mlca replay --verify` re-executes every generated script in a fresh environment built from the recorded package lock, without the model, and compares outputs and metrics with the record.
- **Studies:** one per task that ran on the laptop: S1 run 1, S2 RNA ER run 1, S2 RNA PR run 1, S2 proteome ER run 3. VM runs record VM file paths and cannot be replayed here.
- **Result:** **37 of 37 generated scripts reproduced their recorded outputs and metrics exactly** (no differences).
  - Every split regenerates with identical partition assignments.
  - MLCA's own split check passes on 3 studies. On S1 it reports a mismatch only because, after duplicate removal, the checker renumbers rows 0–533 while the stored split keeps the original row ids. That is a limitation of the checker, not of the split.

## 4. AutoML vs MLCA with bootstrap intervals (`automl_bootstrap/`, for R1-1 and R2-3)

Script `bootstrap.py` → `bootstrap.csv`.
- **Method:** for each of the 46 completed pairs, MLCA's saved test predictions are re-exported (hash-verified) and scored with AutoML's saved predictions on the same test rows. The 95% interval of the difference (AutoML minus MLCA) comes from 2,000 paired bootstrap resamples.
- **Result:** **45 of 46 intervals include zero.** The exception is S1 run 2 with AutoGluon (+0.11, interval 0.01–0.20), where the agent had excluded Hemoglobin and AutoML received the same reduced data.
- **Test-set sizes:** S1 108, S2 25–26, S3 201. The wide S2 intervals (up to ±0.15) show how little the small test sets can resolve.
- **Reading:** comparable discrimination. No claim of superiority in either direction.

## 5. Cost per phase and automatic recovery (`phases/`, for R1-4 and R2-2)

Script `phases.py` → `phases.csv`, `recovery.csv`, `SUMMARY.md`. The cost covers every request of every agent run, including retries and failed attempts. DeepSeek API requests are priced as in section 1. OpenRouter requests (development batches 1–8) use the cost OpenRouter returned with each response; every request had one.

**Whole study: US$ 17.47**:
| Phase | Cost |
|---|---:|
| development | $6.89 |
| acceptance | $4.06 |
| held-out, rails 0.13 | $1.44 |
| first biomedical run | $0.64 |
| development after the reopening | $1.61 |
| biomedical evaluation, released (all attempts in its folders) | $2.14 |
| held-out, released | $0.69 |

**Automatic recovery:** a step counts as recovered when it completed after at least one failed attempt, through MLCA's own repair. Recovery rate = recovered / (recovered + failed).
- **Released biomedical evaluation:** 291 of 317 steps completed at the first attempt; 16 of 26 failing steps were recovered (62%).
- **Released held-out tables:** 122 of 135 at the first attempt; 12 of 13 recovered (92%).
- **Development:** the rate rose from 27% (batch 1) to 100% (batches 11–13).

## 6. Held-out tables on the released version (`heldout02/`)

Script `heldout02.py` → `decisions.csv`, `bootstrap.csv`.

**Outcomes:** 6/12 successful (Bank Marketing 3/3, Wholesale 3/3, Steel 0/3, Wine 0/3). 11 of 12 studies completed.

**All five completed failures inferred the declared targets but a different problem type:**
- Steel: multi-label over the seven fault columns, where a 7-class label was declared;
- Wine: 7-class classification of the 3–9 quality grade, where regression was declared.

**Steel run 2** stopped at the data step. Before any split, the agent tried to record class shares as metrics:
- first with partition "all", refused with the hint to use fact candidates;
- then relabelled "external", refused by the test lock ("External metric is not permitted").

**AutoML on the same split:**
- Bank Marketing (test n = 9,044): AutoML higher by 0.011 ROC AUC (95% interval about 0.004–0.019; MLCA 0.777, AutoGluon 0.789, FLAML 0.788).
- Wine: AutoML lower RMSE (0.69 vs 0.81). But MLCA solved it as classification, so its RMSE is computed on class predictions. This is not a like-for-like comparison.
- Steel: no MLCA metric (multi-label).
- Wholesale: no AutoML counterpart (clustering).

As in the biomedical tasks, the three runs of a table share one split.
