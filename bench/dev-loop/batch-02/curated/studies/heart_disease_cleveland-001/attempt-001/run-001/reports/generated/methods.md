# Methods draft

Draft generated from the event log. Author review is required.
Agent file access: unknown (outside MLCA).

Workflow: tabular_classification 0.3.0
Measurement arm: worker
Task: {'type': 'binary', 'primary_metric': 'roc_auc', 'metric_reason': '', 'targets': ['heart_disease_binary'], 'time_columns': [], 'autonomous': True, 'n_classes': 2}
Resolved columns: {'target': 'heart_disease_binary'}
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
Split manifest SHA256: 7bae94d9929b3004d45c0b6bb5ff30790ba5a4718f35ea246baba4f8eb48d28b

## 01_onboard_confirm_brief_G21F3PYA
The study aim, target, unit of analysis and data inventory were recorded before analysis.
Parameters: {}

## 02_data_audit_schema_Y3QC3VCX
Column types, unique values and missingness were audited on the registered table.
Parameters: {}

## 03_eda_describe_distributions_PGCDWA0V
Distributions were summarised before splitting; no transformation was fitted.
Parameters: {}

## 04_split_freeze_stratified_B8KGKW9G
Partitions (60/20/20) were frozen stratified by the target with a fixed seed; the test partition was encrypted.
Parameters: {'method': 'stratified', 'train_fraction': 0.6, 'val_fraction': 0.2, 'seed': 42, 'group_column': '', 'time_column': ''}

## 05_preprocess_fit_preprocessing_XHDQ4AZ4
Median/most-frequent imputation, one-hot encoding and standard scaling were fitted on training rows only.
Parameters: {}

## 06_features_select_features_ZYCM1WYT
Features were selected with a mutual-information filter fitted on training rows.
Parameters: {}

## 07_model_train_logistic_D8PWVNVS
A class-balanced L2 logistic regression was fitted on training rows as the baseline.
Parameters: {}

## 08_model_train_alternative_V3H8V8QN
Histogram gradient boosting with default parameters was fitted on training rows as the prespecified alternative.
Parameters: {}

## 09_validate_score_validation_24S6SA3P
Candidates were compared on validation rows by ROC AUC; the threshold was set by Youden's J on validation.
Parameters: {}

## 10_test_score_test_YJ5KEKMP
The frozen model and threshold were evaluated once on the locked test partition with bootstrap confidence intervals.
Parameters: {}

## 11_interpret_explain_model_DWESY9QZ
Feature importance was estimated by permutation on validation rows.
Parameters: {}

