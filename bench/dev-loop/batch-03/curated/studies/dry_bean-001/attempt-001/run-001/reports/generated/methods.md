# Methods draft

Draft generated from the event log. Author review is required.
Agent file access: unknown (outside MLCA).

Workflow: tabular_classification 0.3.0
Measurement arm: worker
Task: {'type': 'multiclass', 'primary_metric': 'roc_auc_ovr', 'metric_reason': 'Leave primary_metric null per the workflow default; the task is multiclass classification, so no numeric-outcome metric override applies.', 'targets': ['Class'], 'time_columns': [], 'autonomous': True, 'n_classes': 7}
Resolved columns: {'target': 'Class'}
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
Split manifest SHA256: 5810bfd8c1c92712224c1bbe50b0e4929a7270c9a16cdecaffa3e7ea9228c0a4

## Pre-split data decisions

- multiclass: {'kind': 'set_task', 'mode': 'multiclass'} (actor: user:policy/6de754a45507)
- duplicates: {'kind': 'drop_duplicates'} (actor: user:policy/6de754a45507)

## 01_onboard_confirm_brief_61GW3CBT
The study aim, target, unit of analysis and data inventory were recorded before analysis.
Parameters: {}

## 02_data_audit_schema_7SB7C7C7
Column types, unique values and missingness were audited on the registered table.
Parameters: {}

## 03_eda_describe_distributions_BJXFK6QY
Distributions were summarised before splitting; no transformation was fitted.
Parameters: {}

## 04_split_freeze_stratified_08VT1SVX
Partitions (60/20/20) were frozen stratified by the target with a fixed seed; the test partition was encrypted.
Parameters: {'method': 'stratified', 'train_fraction': 0.6, 'val_fraction': 0.2, 'seed': 42, 'group_column': '', 'time_column': ''}

## 05_preprocess_fit_preprocessing_Y4SGGFB6
Median/most-frequent imputation, one-hot encoding and standard scaling were fitted on training rows only.
Parameters: {}

## 06_features_select_features_W4621TGT
Features were selected with a mutual-information filter fitted on training rows.
Parameters: {}

## 07_model_train_logistic_SNK2GEND
A class-balanced L2 logistic regression was fitted on training rows as the baseline.
Parameters: {}

## 08_model_train_alternative_CWV9X77F
Histogram gradient boosting with default parameters was fitted on training rows as the prespecified alternative.
Parameters: {}

## 09_validate_score_validation_F6APQP74
Candidates were compared on validation rows by the declared task-specific primary metric. A binary threshold used validation Youden J; multiclass evaluation used macro one-vs-rest AUC without a binary threshold.
Parameters: {}

## 10_test_score_test_61QZRS8G
The validation-selected frozen model was evaluated once on test rows by the declared task-specific primary metric with bootstrap confidence intervals; binary thresholds stayed frozen and multiclass scoring used macro one-vs-rest AUC.
Parameters: {}

## 11_interpret_explain_model_0955M24T
Permutation feature importance used validation rows and scoring compatible with the declared task-specific primary metric; no model was refitted.
Parameters: {}

