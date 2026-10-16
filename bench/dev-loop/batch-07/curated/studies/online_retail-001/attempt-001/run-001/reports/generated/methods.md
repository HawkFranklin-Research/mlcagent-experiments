# Methods draft

Draft generated from the event log. Author review is required.
Agent file access: unknown (outside MLCA).

Workflow: clustering 0.1.0
Measurement arm: worker
Task: {'type': 'clustering', 'primary_metric': 'silhouette', 'metric_reason': 'No column is a plausible outcome, so the natural unsupervised aim is to segment the transaction/entity rows, evaluated with silhouette.', 'targets': [], 'time_columns': ['InvoiceDate'], 'autonomous': True}
Resolved columns: {}
Environment lock: e35e6f90bb62171d5f85844434b8b3a9181b1ea9ff2655c161fac388f4975c94

Recorded packages:
cloudpickle==3.1.2
contourpy==1.4.0
cycler==0.12.1
fonttools==4.67.0
joblib==1.6.0
kiwisolver==1.5.1
matplotlib==3.11.2
narwhals==2.27.1
numpy==2.5.4
packaging==26.3
pandas==3.0.6
pillow==12.3.0
pyarrow==26.0.0
pyparsing==3.3.3
python-dateutil==2.9.0.post0
scikit-learn==1.9.1
scipy==1.18.1
seaborn==0.13.2
six==1.17.0
threadpoolctl==3.7.0

Frozen split parameters:
{'method': 'group_time', 'train_fraction': 0.6, 'val_fraction': 0.2, 'seed': 42, 'group_column': 'CustomerID', 'time_column': '', 'time_columns': ['InvoiceDate']}
Split manifest SHA256: b5320229c5bce4907e017d2bad38610d437b2773fc54e92ef95782b03fc79bfc

## 01_onboard_confirm_brief_G3YMFSDS
Record brief and data inventory: record the brief answers, resolved columns and file inventory.
Parameters: {}

## 02_data_audit_schema_M22EQHME
Audit columns, types and missingness: one row per column: name, dtype, n_unique, n_missing, pct_missing.
Parameters: {}

## 03_eda_build_entities_PMXXXB63
Build entity-level features: if the registered table contains repeated transactions, create one row per entity using deterministic aggregation (recency, frequency and monetary summaries as appropriate), with no fitted transform and no outcome. Preserve an explicit entity identifier locally; do not print raw records.
Parameters: {}

## 04_split_freeze_group_SB006Y14
Group-disjoint split (no subject in two partitions): engine split with method=group and the given group column.
Parameters: {'method': 'group', 'train_fraction': 0.6, 'val_fraction': 0.2, 'seed': 42, 'group_column': 'CustomerID', 'time_column': ''}

## 05_preprocess_fit_preprocessing_X6XZMNTX
Fit imputation, encoding and scaling: fit a ColumnTransformer using only training feature columns, excluding every target and identifier named in _protocol. Use median imputation, numeric scaling and categorical encoding as needed.
Parameters: {}

## 06_features_retain_features_79PZREQ1
Keep all features (no selection): fit a passthrough selector with fit_record on the full frozen training index; preserve feature names and order.
Parameters: {}

## 07_model_train_baseline_7R0X4ZSD
Fit baseline: fit complete preprocessing + KMeans candidates for a bounded cluster-count range using fit_record on the full frozen training index. Choose cluster count using internal validation metrics and training-only stability, never labels. For bootstrap stability, use sample-weight multiplicities on the full training index; do not fit subsets or relabel rows.
Parameters: {}

## 08_model_train_alternative_3C6W5R97
Fit alternative: fit complete preprocessing + MiniBatchKMeans candidates for a bounded cluster-count range using fit_record on the full frozen training index. Choose cluster count using internal validation metrics and training-only stability, never labels. For bootstrap stability, use sample-weight multiplicities on the full training index; do not fit subsets or relabel rows.
Parameters: {}

## 09_validate_score_validation_T9RQ4KJY
Score candidates on validation rows: Score the frozen candidates using the task-specific declared primary metric and its direction; supplementary metrics do not replace it. Fit only on training rows, select on validation, and evaluate the frozen choice on test.
Parameters: {}

## 10_test_score_test_FNGG9N07
Score the frozen model on test rows: Score the frozen candidates using the task-specific declared primary metric and its direction; supplementary metrics do not replace it. Fit only on training rows, select on validation, and evaluate the frozen choice on test.
Parameters: {}

## 11_interpret_explain_model_XMSY1B3W
Rank features of the frozen model: describe the frozen model, its selected hyperparameters, inputs, validation evidence and limitations, without fitting.
Parameters: {}

