# Methods draft

Draft generated from the event log. Author review is required.
Agent file access: unknown (outside MLCA).

Workflow: tabular_classification 0.3.0
Measurement arm: worker
Task: {'type': 'multiclass', 'primary_metric': 'roc_auc_ovr', 'metric_reason': 'Leave the workflow default metric; the table was assembled for a 7-class forest cover type prediction task, evaluated with the standard multiclass workflow metric.', 'targets': ['Cover_Type'], 'time_columns': [], 'autonomous': True, 'n_classes': 7}
Resolved columns: {'target': 'Cover_Type'}
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
Split manifest SHA256: c9fbfb9ae46cdd6e062022e9a54709ba11a3d74e83e7d09572def5e2de18a1a5

## Pre-split data decisions

- multiclass: {'kind': 'set_task', 'mode': 'multiclass'} (actor: user:policy/fd4344ccf324)
- rare_classes: {'kind': 'keep'} (actor: user:policy/fd4344ccf324)

## 01_onboard_confirm_brief_PTX3AFYV
The study aim, target, unit of analysis and data inventory were recorded before analysis.
Parameters: {}

## 02_data_audit_schema_QYTK8C8S
Column types, unique values and missingness were audited on the registered table.
Parameters: {}

## 03_eda_describe_distributions_24MHG96Y
Distributions were summarised before splitting; no transformation was fitted.
Parameters: {}

## 04_split_freeze_stratified_EDG1VHYE
Partitions (60/20/20) were frozen stratified by the target with a fixed seed; the test partition was encrypted.
Parameters: {'method': 'stratified', 'train_fraction': 0.6, 'val_fraction': 0.2, 'seed': 42, 'group_column': '', 'time_column': ''}

## 05_preprocess_fit_preprocessing_0FJ6AMWD
Median/most-frequent imputation, one-hot encoding and standard scaling were fitted on training rows only.
Parameters: {}

## 06_features_select_features_79Q93RHQ
Features were selected with a mutual-information filter fitted on training rows.
Parameters: {}

## 07_model_train_logistic_JDJV5YWV
A class-balanced L2 logistic regression was fitted on training rows as the baseline.
Parameters: {}

## 08_model_train_alternative_DE6J0FH7
Histogram gradient boosting with default parameters was fitted on training rows as the prespecified alternative.
Parameters: {}

## 09_validate_score_validation_SR5R5AJH
Score candidates on validation rows: Score the frozen candidates using the task-specific declared primary metric and its direction; supplementary metrics do not replace it. Fit only on training rows, select on validation, and evaluate the frozen choice on test.
Parameters: {}

## 10_test_score_test_HP0NX1J2
Score the frozen model on test rows: Score the frozen candidates using the task-specific declared primary metric and its direction; supplementary metrics do not replace it. Fit only on training rows, select on validation, and evaluate the frozen choice on test.
Parameters: {}

## 11_interpret_explain_model_FC1S8X30
Permutation feature importance used validation rows and scoring compatible with the declared task-specific primary metric; no model was refitted.
Parameters: {}

