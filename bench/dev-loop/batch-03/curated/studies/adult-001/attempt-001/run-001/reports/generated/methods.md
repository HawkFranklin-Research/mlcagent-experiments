# Methods draft

Draft generated from the event log. Author review is required.
Agent file access: unknown (outside MLCA).

Workflow: tabular_classification 0.3.0
Measurement arm: worker
Task: {'type': 'binary', 'primary_metric': 'roc_auc', 'metric_reason': 'Workflow metric is used; income is a binary outcome after merging trailing-period label variants, so no numeric-outcome override is needed.', 'targets': ['income'], 'time_columns': [], 'autonomous': True, 'n_classes': 2}
Resolved columns: {'target': 'income'}
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
{'method': 'stratified', 'train_fraction': 0.6, 'val_fraction': 0.2, 'seed': 42, 'group_column': '', 'time_column': ''}
Split manifest SHA256: c3fb4cb33538e0b1f0df210d83b87d13e604659cdae7bbe219c00a3b9a2f68a9

## Pre-split data decisions

- outcome_choice: {'kind': 'set_outcome', 'column': 'income', 'exclude': ['marital-status']} (actor: user:policy/e274a8be2aca)
- duplicates: {'kind': 'drop_duplicates'} (actor: user:policy/e274a8be2aca)

## 01_onboard_confirm_brief_VC5EA0WS
The study aim, target, unit of analysis and data inventory were recorded before analysis.
Parameters: {}

## 02_data_audit_schema_J4VV27PC
Column types, unique values and missingness were audited on the registered table.
Parameters: {}

## 03_eda_describe_distributions_QT0TZRJA
Distributions were summarised before splitting; no transformation was fitted.
Parameters: {}

## 04_split_freeze_stratified_APG5VE9W
Partitions (60/20/20) were frozen stratified by the target with a fixed seed; the test partition was encrypted.
Parameters: {'method': 'stratified', 'train_fraction': 0.6, 'val_fraction': 0.2, 'seed': 42, 'group_column': '', 'time_column': ''}

## 05_preprocess_fit_preprocessing_F5B2KN9D
Median/most-frequent imputation, one-hot encoding and standard scaling were fitted on training rows only.
Parameters: {}

## 06_features_select_features_Y4ZJEG70
Features were selected with a mutual-information filter fitted on training rows.
Parameters: {}

## 07_model_train_logistic_CV1F3546
A class-balanced L2 logistic regression was fitted on training rows as the baseline.
Parameters: {}

## 08_model_train_alternative_Z4FMHBXH
Histogram gradient boosting with default parameters was fitted on training rows as the prespecified alternative.
Parameters: {}

## 09_validate_score_validation_N6C8NZXN
Candidates were compared on validation rows by the declared task-specific primary metric. A binary threshold used validation Youden J; multiclass evaluation used macro one-vs-rest AUC without a binary threshold.
Parameters: {}

## 10_test_score_test_FQRE0PDX
The validation-selected frozen model was evaluated once on test rows by the declared task-specific primary metric with bootstrap confidence intervals; binary thresholds stayed frozen and multiclass scoring used macro one-vs-rest AUC.
Parameters: {}

## 11_interpret_explain_model_N7R95FD7
Permutation feature importance used validation rows and scoring compatible with the declared task-specific primary metric; no model was refitted.
Parameters: {}

