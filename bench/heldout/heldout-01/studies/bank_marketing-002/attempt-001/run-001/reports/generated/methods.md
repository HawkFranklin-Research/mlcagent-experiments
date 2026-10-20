# Methods draft

Draft generated from the event log. Author review is required.
Agent file access: unknown (outside MLCA).

Workflow: tabular_classification 0.3.0
Measurement arm: worker
Task: {'type': 'binary', 'primary_metric': 'roc_auc', 'metric_reason': 'Workflow metric left null for binary classification; the harness default ranking metric applies.', 'targets': ['y'], 'time_columns': [], 'autonomous': True, 'n_classes': 2}
Resolved columns: {'target': 'y'}
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
{'method': 'stratified', 'train_fraction': 0.6, 'val_fraction': 0.2, 'seed': 42, 'group_column': '', 'time_column': ''}
Split manifest SHA256: 0a509bca9ea01f95ecfb2e6f1d3e0849d86359c1e1926d18c14dcf2e26c072ff

## Pre-split data decisions

- leakage:day: {'kind': 'exclude_column', 'column': 'day'} (actor: user:policy/80796950960d)
- leakage:pdays: {'kind': 'exclude_column', 'column': 'pdays'} (actor: user:policy/80796950960d)

## 01_onboard_confirm_brief_JP5AW86Z
The study aim, target, unit of analysis and data inventory were recorded before analysis.
Parameters: {}

## 02_data_count_target_CZH5PK5E
Target classes were counted.
Parameters: {}

## 03_data_audit_schema_PPPKYC7T
Column types, unique values and missingness were audited on the registered table.
Parameters: {}

## 04_eda_describe_distributions_4J53P60B
Distributions were summarised before splitting; no transformation was fitted.
Parameters: {}

## 05_split_freeze_stratified_WDC0Z3NE
Partitions (60/20/20) were frozen stratified by the target with a fixed seed; the test partition was encrypted.
Parameters: {'method': 'stratified', 'train_fraction': 0.6, 'val_fraction': 0.2, 'seed': 42, 'group_column': '', 'time_column': ''}

## 06_preprocess_fit_preprocessing_STWKFGVD
Median/most-frequent imputation, one-hot encoding and standard scaling were fitted on training rows only.
Parameters: {}

## 07_features_select_features_N6F5BDYX
Features were selected with a mutual-information filter fitted on training rows.
Parameters: {}

## 08_model_train_logistic_Y7873DWF
A class-balanced L2 logistic regression was fitted on training rows as the baseline.
Parameters: {}

## 09_model_train_alternative_F9H6TZDH
Histogram gradient boosting with default parameters was fitted on training rows as the prespecified alternative.
Parameters: {}

## 10_validate_score_validation_7W7VKTVE
Score candidates on validation rows: Score the frozen candidates using the task-specific declared primary metric and its direction; supplementary metrics do not replace it. Fit only on training rows, select on validation, and evaluate the frozen choice on test.
Parameters: {}

## 11_validate_score_validation_5PKKGYTK
Score candidates on validation rows: Score the frozen candidates using the task-specific declared primary metric and its direction; supplementary metrics do not replace it. Fit only on training rows, select on validation, and evaluate the frozen choice on test.
Parameters: {}

## 12_test_score_test_X80AWWYE
Score the frozen model on test rows: Score the frozen candidates using the task-specific declared primary metric and its direction; supplementary metrics do not replace it. Fit only on training rows, select on validation, and evaluate the frozen choice on test. Per-row test predictions are saved locally for external paired comparison.
Parameters: {}

## 13_interpret_explain_model_NSAAVTXF
Permutation feature importance used validation rows and scoring compatible with the declared task-specific primary metric; no model was refitted.
Parameters: {}

