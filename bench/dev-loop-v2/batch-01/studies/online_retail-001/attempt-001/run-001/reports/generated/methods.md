# Methods draft

Draft generated from the event log. Author review is required.
Agent file access: unknown (outside MLCA).

Workflow: clustering 0.1.0
Measurement arm: worker
Task: {'type': 'clustering', 'primary_metric': 'silhouette', 'metric_reason': 'Workflow default retained: clustering tasks have no held-out label, so the pipeline supplies the workflow metric.', 'targets': [], 'time_columns': [], 'autonomous': True}
Resolved columns: {}
Environment lock: 3e677b448fb005ba15ca050b73c44d764f1292910aaddc9c0e85e3c55bb7d639

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
pillow==13.0.0
pyarrow==26.0.0
pyparsing==3.3.3
python-dateutil==2.9.0.post0
scikit-learn==1.9.1
scipy==1.18.1
seaborn==0.13.2
six==1.17.0
threadpoolctl==3.7.0

Frozen split parameters:
{'method': 'group', 'train_fraction': 0.6, 'val_fraction': 0.2, 'seed': 42, 'group_column': 'CustomerID', 'time_column': ''}
Split manifest SHA256: 161c9661818f8e6d70aa6409b06644ebf3b0a424b811c420fc93db2c7a3dd84d

## 01_onboard_confirm_brief_0N7XK1AW
Record brief and data inventory: record the brief answers, resolved columns and file inventory.
Parameters: {}

## 02_data_audit_schema_G7HTQ7ZR
Audit columns, types and missingness: one row per column: name, dtype, n_unique, n_missing, pct_missing.
Parameters: {}

## 03_eda_build_entities_QKYRW64C
Build entity-level features: if the registered table contains repeated transactions, create one row per entity using deterministic aggregation (recency, frequency and monetary summaries as appropriate), with no fitted transform and no outcome. Preserve an explicit entity identifier locally; do not print raw records.
Parameters: {}

## 04_split_freeze_group_DKY03W5P
Group-disjoint split (no subject in two partitions): engine split with method=group and the given group column.
Parameters: {'method': 'group', 'train_fraction': 0.6, 'val_fraction': 0.2, 'seed': 42, 'group_column': 'CustomerID', 'time_column': ''}

## 05_preprocess_fit_preprocessing_8WCJT1S7
Fit imputation, encoding and scaling: fit a ColumnTransformer using only training feature columns, excluding every target and identifier named in _protocol. Use median imputation, numeric scaling and categorical encoding as needed.
Parameters: {}

## 06_features_retain_features_2NP0WC88
Keep all features (no selection): fit a passthrough selector with fit_record on the full frozen training index; preserve feature names and order.
Parameters: {}

## 07_model_train_baseline_XYVYQTXX
Fit baseline: fit complete preprocessing + KMeans candidates for a bounded cluster-count range using fit_record on the full frozen training index. Choose cluster count using internal validation metrics and training-only stability, never labels. For bootstrap stability, use sample-weight multiplicities on the full training index; do not fit subsets or relabel rows.
Parameters: {}

## 08_model_train_alternative_RH7MFB6Q
Fit alternative: fit complete preprocessing + MiniBatchKMeans candidates for a bounded cluster-count range using fit_record on the full frozen training index. Choose cluster count using internal validation metrics and training-only stability, never labels. For bootstrap stability, use sample-weight multiplicities on the full training index; do not fit subsets or relabel rows.
Parameters: {}

## 09_validate_score_validation_TZGPKJKF
Score candidates on validation rows: Score the frozen candidates using the task-specific declared primary metric and its direction; supplementary metrics do not replace it. Fit only on training rows, select on validation, and evaluate the frozen choice on test.
Parameters: {}

## 10_test_score_test_QE6B1RKE
Score the frozen model on test rows: Score the frozen candidates using the task-specific declared primary metric and its direction; supplementary metrics do not replace it. Fit only on training rows, select on validation, and evaluate the frozen choice on test.
Parameters: {}

## 11_interpret_explain_model_J3EXTDR6
Rank features of the frozen model: describe the frozen model, its selected hyperparameters, inputs, validation evidence and limitations, without fitting.
Parameters: {}

