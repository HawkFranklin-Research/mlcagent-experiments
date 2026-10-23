# Methods draft

Draft generated from the event log. Author review is required.
Agent file access: unknown (outside MLCA).

Workflow: tabular_classification 0.3.0
Measurement arm: worker
Task: {'type': 'binary', 'primary_metric': 'roc_auc', 'metric_reason': 'Binary classification uses the workflow-selected default metric; no numeric-outcome override applies.', 'targets': ['cancer_status'], 'time_columns': [], 'autonomous': True, 'n_classes': 2}
Resolved columns: {'target': 'cancer_status'}
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
Split manifest SHA256: 4ecb0c5e8cb20b97a39650f626fb17fd512ecf8863f34df164c64129be8191f1

## 01_onboard_confirm_brief_ZS06Q4BV
The study aim, target, unit of analysis and data inventory were recorded before analysis.
Parameters: {}

## 02_data_count_target_TKK7BPBS
Target classes were counted.
Parameters: {}

## 03_data_audit_schema_PYAX6JDZ
Column types, unique values and missingness were audited on the registered table.
Parameters: {}

## 04_eda_describe_distributions_ZBXF9ZFM
Distributions were summarised before splitting; no transformation was fitted.
Parameters: {}

## 05_split_freeze_stratified_9S5G0BEZ
Partitions (60/20/20) were frozen stratified by the target with a fixed seed; the test partition was encrypted.
Parameters: {'method': 'stratified', 'train_fraction': 0.6, 'val_fraction': 0.2, 'seed': 42, 'group_column': '', 'time_column': ''}

## 06_preprocess_fit_preprocessing_XWW5WTFN
Median/most-frequent imputation, one-hot encoding and standard scaling were fitted on training rows only.
Parameters: {}

## 07_features_select_features_3JX8FX9S
Features were selected with a mutual-information filter fitted on training rows.
Parameters: {}

## 08_model_train_logistic_1JG6Z9GW
A class-balanced L2 logistic regression was fitted on training rows as the baseline.
Parameters: {}

## 09_model_train_alternative_N0C35K8C
Histogram gradient boosting with default parameters was fitted on training rows as the prespecified alternative.
Parameters: {}

## 10_validate_score_validation_EHS3PS8T
Score candidates on validation rows: Score the frozen candidates using the task-specific declared primary metric and its direction; supplementary metrics do not replace it. Fit only on training rows, select on validation, and evaluate the frozen choice on test.
Parameters: {}

## 11_test_score_test_XHM4HGW0
Score the frozen model on test rows: Score the frozen candidates using the task-specific declared primary metric and its direction; supplementary metrics do not replace it. Fit only on training rows, select on validation, and evaluate the frozen choice on test. Per-row test predictions are saved locally for external paired comparison.
Parameters: {}

## 12_interpret_explain_model_67271NZ2
Feature ranking used the recorded native model importances or validation permutation method and budget. Above 500 input features, permutation is limited to the top 50 ranked by native importances and at most 5 repeats; no model is refitted.
Parameters: {}

