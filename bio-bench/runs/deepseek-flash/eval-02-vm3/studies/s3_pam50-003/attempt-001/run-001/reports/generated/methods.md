# Methods draft

Draft generated from the event log. Author review is required.
Agent file access: unknown (outside MLCA).

Workflow: tabular_classification 0.3.0
Measurement arm: worker
Task: {'type': 'multiclass', 'primary_metric': 'roc_auc_ovr', 'metric_reason': '', 'targets': ['PAM50'], 'time_columns': [], 'autonomous': True, 'n_classes': 5}
Resolved columns: {'target': 'PAM50'}
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
Split manifest SHA256: 8553e182743710cfc772b31bd95fbd3b3574c08e202ae6c721ed9c7abc485e43

## Pre-split data decisions

- multiclass: {'kind': 'set_task', 'mode': 'multiclass'} (actor: user:policy/c48ac21519e3)
- identifier:sample_id: {'kind': 'exclude_column', 'column': 'sample_id'} (actor: user:policy/c48ac21519e3)

## 01_onboard_confirm_brief_JZT2EWWN
The study aim, target, unit of analysis and data inventory were recorded before analysis.
Parameters: {}

## 02_data_count_target_QTZ33S3Q
Target classes were counted.
Parameters: {}

## 03_data_audit_schema_6Q8GGP2B
Column types, unique values and missingness were audited on the registered table.
Parameters: {}

## 04_eda_describe_distributions_CX465F96
Distributions were summarised before splitting; no transformation was fitted.
Parameters: {}

## 05_split_freeze_stratified_X1AQA0VT
Partitions (60/20/20) were frozen stratified by the target with a fixed seed; the test partition was encrypted.
Parameters: {'method': 'stratified', 'train_fraction': 0.6, 'val_fraction': 0.2, 'seed': 42, 'group_column': '', 'time_column': ''}

## 06_preprocess_fit_preprocessing_6C1GR31N
Median/most-frequent imputation, one-hot encoding and standard scaling were fitted on training rows only.
Parameters: {}

## 07_features_select_features_2XVWGVRB
Features were selected with a mutual-information filter fitted on training rows.
Parameters: {}

## 08_model_train_logistic_7JTSZ4DZ
A class-balanced L2 logistic regression was fitted on training rows as the baseline.
Parameters: {}

## 09_model_train_alternative_94QWT4VC
Histogram gradient boosting with default parameters was fitted on training rows as the prespecified alternative.
Parameters: {}

## 10_validate_score_validation_6TSKNNH9
Score candidates on validation rows: Score the frozen candidates using the task-specific declared primary metric and its direction; supplementary metrics do not replace it. Fit only on training rows, select on validation, and evaluate the frozen choice on test.
Parameters: {}

## 11_test_score_test_7HW6NN8K
Score the frozen model on test rows: Score the frozen candidates using the task-specific declared primary metric and its direction; supplementary metrics do not replace it. Fit only on training rows, select on validation, and evaluate the frozen choice on test. Per-row test predictions are saved locally for external paired comparison.
Parameters: {}

## 12_interpret_explain_model_WJT74EXR
Feature ranking used the recorded native model importances or validation permutation method and budget. Above 500 input features, permutation is limited to the top 50 ranked by native importances and at most 5 repeats; no model is refitted.
Parameters: {}

