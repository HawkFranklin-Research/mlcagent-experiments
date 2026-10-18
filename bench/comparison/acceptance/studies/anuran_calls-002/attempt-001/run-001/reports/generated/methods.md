# Methods draft

Draft generated from the event log. Author review is required.
Agent file access: unknown (outside MLCA).

Workflow: tabular_classification 0.3.0
Measurement arm: worker
Task: {'type': 'multiclass', 'primary_metric': 'roc_auc_ovr', 'metric_reason': "Multiclass classification with the workflow's default metric; no numeric outcome applies, so no override is justified.", 'targets': ['Species'], 'time_columns': [], 'autonomous': True, 'n_classes': 10}
Resolved columns: {'target': 'Species'}
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
{'method': 'group', 'train_fraction': 0.6, 'val_fraction': 0.2, 'seed': 42, 'group_column': 'RecordID', 'time_column': ''}
Split manifest SHA256: 370d5c584f83346fc8a947e7653b34bf19a53fb9cfe668c5e15222f1ac28672d

## Pre-split data decisions

- multiclass: {'kind': 'set_task', 'mode': 'multiclass'} (actor: user:policy/464899fa2ad3)
- rare_classes: {'kind': 'keep'} (actor: user:policy/464899fa2ad3)

## 01_onboard_confirm_brief_KK7R338R
The study aim, target, unit of analysis and data inventory were recorded before analysis.
Parameters: {}

## 02_data_count_target_SX46N5SA
Target classes were counted.
Parameters: {}

## 03_data_audit_schema_RG93NB58
Column types, unique values and missingness were audited on the registered table.
Parameters: {}

## 04_eda_describe_distributions_ZM0C11G1
Distributions were summarised before splitting; no transformation was fitted.
Parameters: {}

## 05_split_freeze_group_H45V0FT9
Partitions were frozen group-disjointly with a fixed seed; the test partition was encrypted.
Parameters: {'method': 'group', 'train_fraction': 0.6, 'val_fraction': 0.2, 'seed': 42, 'group_column': 'RecordID', 'time_column': ''}

## 06_preprocess_fit_preprocessing_MYPQXFW6
Median/most-frequent imputation, one-hot encoding and standard scaling were fitted on training rows only.
Parameters: {}

## 07_features_select_features_NK102NGP
Features were selected with a mutual-information filter fitted on training rows.
Parameters: {}

## 08_model_train_logistic_GP5KS1CK
A class-balanced L2 logistic regression was fitted on training rows as the baseline.
Parameters: {}

## 09_model_train_alternative_16WYVMV8
Histogram gradient boosting with default parameters was fitted on training rows as the prespecified alternative.
Parameters: {}

## 10_validate_score_validation_XTD87YEX
Score candidates on validation rows: Score the frozen candidates using the task-specific declared primary metric and its direction; supplementary metrics do not replace it. Fit only on training rows, select on validation, and evaluate the frozen choice on test.
Parameters: {}

## 11_test_score_test_056XMNWZ
Score the frozen model on test rows: Score the frozen candidates using the task-specific declared primary metric and its direction; supplementary metrics do not replace it. Fit only on training rows, select on validation, and evaluate the frozen choice on test.
Parameters: {}

## 12_interpret_explain_model_FSRB26DG
Permutation feature importance used validation rows and scoring compatible with the declared task-specific primary metric; no model was refitted.
Parameters: {}

