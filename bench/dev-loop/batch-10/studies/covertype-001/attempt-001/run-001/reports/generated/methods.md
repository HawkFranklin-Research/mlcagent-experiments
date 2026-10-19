# Methods draft

Draft generated from the event log. Author review is required.
Agent file access: unknown (outside MLCA).

Workflow: tabular_classification 0.3.0
Measurement arm: worker
Task: {'type': 'multiclass', 'primary_metric': 'roc_auc_ovr', 'metric_reason': 'Multiclass outcome with seven cover types; defer to the workflow default metric rather than overriding.', 'targets': ['Cover_Type'], 'time_columns': [], 'autonomous': True, 'n_classes': 7}
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

- multiclass: {'kind': 'set_task', 'mode': 'multiclass'} (actor: user:policy/732bcc0353d7)
- rare_classes: {'kind': 'keep'} (actor: user:policy/732bcc0353d7)

## 01_onboard_confirm_brief_HD6WVVJT
The study aim, target, unit of analysis and data inventory were recorded before analysis.
Parameters: {}

## 02_data_count_target_B889WZ09
Target classes were counted.
Parameters: {}

## 03_data_audit_schema_HGKADGJ0
Column types, unique values and missingness were audited on the registered table.
Parameters: {}

## 04_eda_describe_distributions_P423HNXH
Distributions were summarised before splitting; no transformation was fitted.
Parameters: {}

## 05_split_freeze_stratified_19YEM2JR
Partitions (60/20/20) were frozen stratified by the target with a fixed seed; the test partition was encrypted.
Parameters: {'method': 'stratified', 'train_fraction': 0.6, 'val_fraction': 0.2, 'seed': 42, 'group_column': '', 'time_column': ''}

## 06_preprocess_fit_preprocessing_XZ1WZNJH
Median/most-frequent imputation, one-hot encoding and standard scaling were fitted on training rows only.
Parameters: {}

## 07_features_select_features_KRQ7615Z
Features were selected with a mutual-information filter fitted on training rows.
Parameters: {}

## 08_model_train_logistic_6YJ0NC7Y
A class-balanced L2 logistic regression was fitted on training rows as the baseline.
Parameters: {}

## 09_model_train_alternative_143BSS6M
Histogram gradient boosting with default parameters was fitted on training rows as the prespecified alternative.
Parameters: {}

## 10_validate_score_validation_B4M4NX4R
Score candidates on validation rows: Score the frozen candidates using the task-specific declared primary metric and its direction; supplementary metrics do not replace it. Fit only on training rows, select on validation, and evaluate the frozen choice on test.
Parameters: {}

## 11_validate_score_validation_9RDBMSFC
Score candidates on validation rows: Score the frozen candidates using the task-specific declared primary metric and its direction; supplementary metrics do not replace it. Fit only on training rows, select on validation, and evaluate the frozen choice on test.
Parameters: {}

## 12_test_score_test_GE7QKZQK
Score the frozen model on test rows: Score the frozen candidates using the task-specific declared primary metric and its direction; supplementary metrics do not replace it. Fit only on training rows, select on validation, and evaluate the frozen choice on test.
Parameters: {}

## 13_interpret_explain_model_TTBMK22C
Permutation feature importance used validation rows and scoring compatible with the declared task-specific primary metric; no model was refitted.
Parameters: {}

