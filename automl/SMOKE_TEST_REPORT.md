# AutoML small-sample smoke report

Date: 2026-10-20. This is a code smoke check only; the constructed partitions are not experimental results and are not paired with frozen MLCA studies.

## Method

- All 12 development task definitions were checked against the AutoML implementation's declared scope.
- For the 7 supported single-label binary, multiclass and regression tasks, a bounded prefix of each CSV was read (at most 5,000 rows, except the small Dry Bean file which was read completely). A deterministic, stratified sample of at most 245 rows for classification, or at most 245 rows for regression, was saved to a temporary Parquet input. No held-out or biomedical file was read.
- For each eligible task, a synthetic MLCA event history (including a hash-checked `SplitFrozen` record), sample membership manifest and train/validation/test Parquet partitions were built from that tiny sample, then passed through the real hash-checking exporter. These are smoke fixtures, not frozen study splits. AutoGluon and FLAML then ran sequentially as separate subprocesses, 20-second framework budget, 2 CPUs, seed 421. Each framework ran even if the other failed. Each subprocess had a 180-second wall timeout. The generated summary was also run on the 14 completed result files and produced 14 rows.
- For the five tasks outside the work order's supported scope (multilabel, multioutput regression, clustering and representation), the actual task-spec loader was called. It rejected each with `Unsupported task type`; no associated table file was opened. These were not incorrectly converted to single-target tasks or fitted.
- Output artifacts and logs are under ignored `bench/automl/.work/small-sample-smoke/`; the only additional tracked file from this smoke task is this report. No source fixes were made.

## Results

| Development table | Declared task | Sample rows | AutoGluon | FLAML |
|---|---|---:|---|---|
| `breast_cancer_diagnostic` | binary | 70 rows x 31 cols from bounded source scan (569 rows read) | **PASS**; test roc_auc=1.0; fit 3.8s; wall 5.4s | **PASS**; test roc_auc=1.0; fit 20.7s; wall 22.2s |
| `heart_disease_cleveland` | binary | 70 rows x 15 cols from bounded source scan (303 rows read) | **PASS**; test roc_auc=0.9183673469387755; fit 3.8s; wall 5.4s | **PASS**; test roc_auc=0.9387755102040816; fit 20.3s; wall 21.9s |
| `adult` | binary | 70 rows x 15 cols from bounded source scan (5000 rows read) | **PASS**; test roc_auc=0.8571428571428572; fit 4.2s; wall 5.8s | **PASS**; test roc_auc=0.8163265306122448; fit 20.3s; wall 21.7s |
| `dry_bean` | multiclass | 245 rows x 17 cols from bounded source scan (13611 rows read) | **PASS**; test roc_auc_ovr=0.9951409135082605; fit 19.8s; wall 21.4s | **PASS**; test roc_auc_ovr=0.988824101068999; fit 20.3s; wall 21.8s |
| `covertype` | multiclass | 245 rows x 55 cols from bounded source scan (5000 rows read) | **PASS**; test roc_auc_ovr=0.9504373177842567; fit 12.8s; wall 14.3s | **PASS**; test roc_auc_ovr=0.9489795918367346; fit 20.4s; wall 22.1s |
| `anuran_calls` | multilabel | not opened (out of scope) | NOT RUN (scope rejection: Unsupported task type: multilabel) | NOT RUN (scope rejection: Unsupported task type: multilabel) |
| `abalone` | regression | 245 rows x 9 cols from bounded source scan (4177 rows read) | **PASS**; test rmse=2.6948013484403917; fit 2.8s; wall 4.3s | **PASS**; test rmse=2.9307552826439425; fit 20.3s; wall 21.7s |
| `bike_sharing_hourly` | regression | 245 rows x 17 cols from bounded source scan (5000 rows read) | **PASS**; test rmse=62.80201443885478; fit 20.5s; wall 21.8s | **PASS**; test rmse=65.19438629166447; fit 20.3s; wall 21.7s |
| `energy_efficiency` | multioutput_regression | not opened (out of scope) | NOT RUN (scope rejection: Unsupported task type: multioutput_regression) | NOT RUN (scope rejection: Unsupported task type: multioutput_regression) |
| `seeds` | clustering | not opened (out of scope) | NOT RUN (scope rejection: Unsupported task type: clustering) | NOT RUN (scope rejection: Unsupported task type: clustering) |
| `online_retail` | clustering | not opened (out of scope) | NOT RUN (scope rejection: Unsupported task type: clustering) | NOT RUN (scope rejection: Unsupported task type: clustering) |
| `optical_digits` | representation | not opened (out of scope) | NOT RUN (scope rejection: Unsupported task type: representation) | NOT RUN (scope rejection: Unsupported task type: representation) |

## Per-run details

### breast_cancer_diagnostic

Sample: 70 rows x 31 cols from bounded source scan (569 rows read)

- autogluon: **PASS**; test roc_auc=1.0; fit 3.8s; wall 5.4s
- flaml: **PASS**; test roc_auc=1.0; fit 20.7s; wall 22.2s

### heart_disease_cleveland

Sample: 70 rows x 15 cols from bounded source scan (303 rows read)

- autogluon: **PASS**; test roc_auc=0.9183673469387755; fit 3.8s; wall 5.4s
- flaml: **PASS**; test roc_auc=0.9387755102040816; fit 20.3s; wall 21.9s

### adult

Sample: 70 rows x 15 cols from bounded source scan (5000 rows read)

- autogluon: **PASS**; test roc_auc=0.8571428571428572; fit 4.2s; wall 5.8s
- flaml: **PASS**; test roc_auc=0.8163265306122448; fit 20.3s; wall 21.7s

### dry_bean

Sample: 245 rows x 17 cols from bounded source scan (13611 rows read)

- autogluon: **PASS**; test roc_auc_ovr=0.9951409135082605; fit 19.8s; wall 21.4s
- flaml: **PASS**; test roc_auc_ovr=0.988824101068999; fit 20.3s; wall 21.8s

### covertype

Sample: 245 rows x 55 cols from bounded source scan (5000 rows read)

- autogluon: **PASS**; test roc_auc_ovr=0.9504373177842567; fit 12.8s; wall 14.3s
- flaml: **PASS**; test roc_auc_ovr=0.9489795918367346; fit 20.4s; wall 22.1s

### anuran_calls

Sample: not opened (out of scope)

- autogluon: NOT RUN (scope rejection: Unsupported task type: multilabel)
- flaml: NOT RUN (scope rejection: Unsupported task type: multilabel)

### abalone

Sample: 245 rows x 9 cols from bounded source scan (4177 rows read)

- autogluon: **PASS**; test rmse=2.6948013484403917; fit 2.8s; wall 4.3s
- flaml: **PASS**; test rmse=2.9307552826439425; fit 20.3s; wall 21.7s

### bike_sharing_hourly

Sample: 245 rows x 17 cols from bounded source scan (5000 rows read)

- autogluon: **PASS**; test rmse=62.80201443885478; fit 20.5s; wall 21.8s
- flaml: **PASS**; test rmse=65.19438629166447; fit 20.3s; wall 21.7s

### energy_efficiency

Sample: not opened (out of scope)

- autogluon: NOT RUN (scope rejection: Unsupported task type: multioutput_regression)
- flaml: NOT RUN (scope rejection: Unsupported task type: multioutput_regression)

### seeds

Sample: not opened (out of scope)

- autogluon: NOT RUN (scope rejection: Unsupported task type: clustering)
- flaml: NOT RUN (scope rejection: Unsupported task type: clustering)

### online_retail

Sample: not opened (out of scope)

- autogluon: NOT RUN (scope rejection: Unsupported task type: clustering)
- flaml: NOT RUN (scope rejection: Unsupported task type: clustering)

### optical_digits

Sample: not opened (out of scope)

- autogluon: NOT RUN (scope rejection: Unsupported task type: representation)
- flaml: NOT RUN (scope rejection: Unsupported task type: representation)

## Environment and limitations

AutoGluon 1.5.0, FLAML 2.7.0, scikit-learn 1.7.2, NumPy 2.3.5, pandas 2.3.3, PyArrow 20.0.0, LightGBM 4.6.0, Python 3.13.12. The sample split has only 245 rows maximum and does not measure predictive performance or stability. These sample metrics are not performance evidence. The smoke does not test comparisons against any real frozen MLCA split, original full-table runtime, or biological tasks.

Configuration: 20s framework budget, 2 CPUs, seed 421; per-process hard timeout 180s. All 14 supported framework runs completed with finite test metrics; no fitting errors occurred. FLAML fit times were 20.3–20.7s (slightly above its 20s budget in all seven runs); AutoGluon exceeded 20s on Bike Sharing (20.5s). This confirms the framework time budget is not a strict wall-clock cutoff. The generated summary contained 14 completed rows. Results and subprocess logs remain in the ignored work folder.
