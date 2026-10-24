# Methods draft

Draft generated from the event log. Author review is required.
Agent file access: unknown (outside MLCA).

Workflow: tabular_classification 0.3.0
Measurement arm: worker
Task: {'type': 'binary', 'targets': ['y'], 'primary_metric': 'roc_auc', 'time_columns': []}
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
Split manifest SHA256: ea52a2625babf72004ac0b1e489619db83ce3ed52addd75f694e5443dea70374

## 01_onboard_confirm_brief_YKY1NXVT
The study aim, target, unit of analysis and data inventory were recorded before analysis.
Parameters: {}

## 02_data_audit_schema_H1CPPRKK
Column types, unique values and missingness were audited on the registered table.
Parameters: {}

## 03_eda_describe_distributions_SQKHSEGX
Distributions were summarised before splitting; no transformation was fitted.
Parameters: {}

## 04_split_freeze_stratified_AR34VW8T
Partitions (60/20/20) were frozen stratified by the target with a fixed seed; the test partition was encrypted.
Parameters: {'method': 'stratified', 'train_fraction': 0.6, 'val_fraction': 0.2, 'seed': 42, 'group_column': '', 'time_column': ''}

## 05_preprocess_fit_preprocessing_DKWT815N
Median/most-frequent imputation, one-hot encoding and standard scaling were fitted on training rows only.
Parameters: {}

## 06_features_select_features_3AP5FWYK
Features were selected with a mutual-information filter fitted on training rows.
Parameters: {}

## 07_model_train_logistic_9WK4YTYW
A class-balanced L2 logistic regression was fitted on training rows as the baseline.
Parameters: {}

## 08_validate_score_validation_NFD33SC6
Score candidates on validation rows: Score the frozen candidates using the task-specific declared primary metric and its direction; supplementary metrics do not replace it. Fit only on training rows, select on validation, and evaluate the frozen choice on test.
Parameters: {}

## 09_test_score_test_HXFEX7ZZ
Score the frozen model on test rows: Score the frozen candidates using the task-specific declared primary metric and its direction; supplementary metrics do not replace it. Fit only on training rows, select on validation, and evaluate the frozen choice on test. Per-row test predictions are saved locally for external paired comparison.
Parameters: {}

## 10_interpret_explain_model_6FEYVBT1
Feature ranking used the recorded native model importances or validation permutation method and budget. Above 500 input features, permutation is limited to the top 50 ranked by native importances and at most 5 repeats; no model is refitted.
Parameters: {}

