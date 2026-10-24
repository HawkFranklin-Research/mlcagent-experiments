# Methods draft

Draft generated from the event log. Author review is required.
Agent file access: unknown (outside MLCA).

Workflow: tabular_classification 0.3.0
Measurement arm: worker
Task: {'type': 'binary', 'primary_metric': 'roc_auc', 'metric_reason': 'Binary outcome task, so the workflow default metric (ROC AUC for discrimination) applies; the study also reports calibration curves, learning curve and SHAP attributions as requested, so no override is needed.', 'targets': ['Result'], 'time_columns': [], 'autonomous': True, 'n_classes': 2}
Resolved columns: {'target': 'Result'}
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
Split manifest SHA256: 7ae90cb5de16a758f3ca35b12013e54c542cc239289c37dbdccefc9b9a178628

## Pre-split data decisions

- duplicates: {'kind': 'drop_duplicates'} (actor: user:policy/d86fc913d628)

## 01_onboard_confirm_brief_ZR4AKMR2
The study aim, target, unit of analysis and data inventory were recorded before analysis.
Parameters: {}

## 02_data_audit_schema_N4KDCH4Y
Column types, unique values and missingness were audited on the registered table.
Parameters: {}

## 03_eda_describe_distributions_ZMN09EVP
Distributions were summarised before splitting; no transformation was fitted.
Parameters: {}

## 04_split_freeze_stratified_0KSHQ3QE
Partitions (60/20/20) were frozen stratified by the target with a fixed seed; the test partition was encrypted.
Parameters: {'method': 'stratified', 'train_fraction': 0.6, 'val_fraction': 0.2, 'seed': 42, 'group_column': '', 'time_column': ''}

## 05_preprocess_fit_preprocessing_3WJD77DF
Median/most-frequent imputation, one-hot encoding and standard scaling were fitted on training rows only.
Parameters: {}

## 06_features_retain_features_6D2RD7T4
All features were retained.
Parameters: {}

## 07_model_train_logistic_SGR96899
A class-balanced L2 logistic regression was fitted on training rows as the baseline.
Parameters: {}

## 08_model_train_alternative_71A7RSCC
Histogram gradient boosting with default parameters was fitted on training rows as the prespecified alternative.
Parameters: {}

## 09_validate_score_validation_J88R8Z29
Score candidates on validation rows: Score the frozen candidates using the task-specific declared primary metric and its direction; supplementary metrics do not replace it. Fit only on training rows, select on validation, and evaluate the frozen choice on test.
Parameters: {}

## 10_test_score_test_4K1EES73
Score the frozen model on test rows: Score the frozen candidates using the task-specific declared primary metric and its direction; supplementary metrics do not replace it. Fit only on training rows, select on validation, and evaluate the frozen choice on test. Per-row test predictions are saved locally for external paired comparison.
Parameters: {}

## 11_interpret_explain_model_HB32E3GS
Feature ranking used the recorded native model importances or validation permutation method and budget. Above 500 input features, permutation is limited to the top 50 ranked by native importances and at most 5 repeats; no model is refitted.
Parameters: {}

