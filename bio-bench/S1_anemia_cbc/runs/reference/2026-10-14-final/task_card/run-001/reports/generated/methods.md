# Methods draft

Draft generated from the event log. Author review is required.
Agent file access: unknown (outside MLCA).

Workflow: tabular_classification 0.3.0
Measurement arm: reference
Task: {'type': 'binary', 'n_classes': 2, 'primary_metric': 'roc_auc'}
Resolved columns: {'target': 'Result'}
Environment lock: b34655987bd4ae46a0af9fe9146953ea30d2fe3e5b4935edd19d8a9f46b23514

No-AI reference comparison: frozen engine-owned scripts; no model inference.
Classification families and validation selection are prespecified.

Recorded packages:
cloudpickle==3.1.2
contourpy==1.4.0
cycler==0.12.1
fonttools==4.66.1
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
Split manifest SHA256: 128afbad2e934a46fbe1f7f54883fa755aa7dbb6ab2e13701d0f9528ab636575

## Pre-split data decisions

- duplicates: {'kind': 'drop_duplicates'} (actor: user:policy/2d792597d4a8)

## 01_onboard_confirm_brief_XTD5BK84
The study aim, target, unit of analysis and data inventory were recorded before analysis.
Parameters: {}

## 02_data_audit_schema_XYZR5ZZN
Column types, unique values and missingness were audited on the registered table.
Parameters: {}

## 03_eda_describe_distributions_JKTHK636
Distributions were summarised before splitting; no transformation was fitted.
Parameters: {}

## 04_split_freeze_stratified_WFMX1HKK
Partitions (60/20/20) were frozen stratified by the target with a fixed seed; the test partition was encrypted.
Parameters: {'method': 'stratified', 'train_fraction': 0.6, 'val_fraction': 0.2, 'seed': 42, 'group_column': '', 'time_column': ''}

## 05_preprocess_fit_preprocessing_C4F41THQ
Median/most-frequent imputation, one-hot encoding and standard scaling were fitted on training rows only.
Parameters: {}

## 06_features_retain_features_R5GJD3J2
All features were retained.
Parameters: {}

## 07_model_train_logistic_AENS9E90
A class-balanced L2 logistic regression was fitted on training rows as the baseline.
Parameters: {}

## 08_model_train_alternative_9ZN3EGZC
Histogram gradient boosting with default parameters was fitted on training rows as the prespecified alternative.
Parameters: {}

## 09_validate_score_validation_F9261W5P
Candidates were compared on validation rows by ROC AUC; the threshold was set by Youden's J on validation.
Parameters: {}

## 10_test_score_test_0A5PVN16
The frozen model and threshold were evaluated once on the locked test partition with bootstrap confidence intervals.
Parameters: {}

## 11_interpret_explain_model_TYVKS62R
Feature importance was estimated by permutation on validation rows.
Parameters: {}

