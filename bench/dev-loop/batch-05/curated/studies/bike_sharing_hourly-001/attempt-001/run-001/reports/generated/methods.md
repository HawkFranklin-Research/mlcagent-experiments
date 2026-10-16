# Methods draft

Draft generated from the event log. Author review is required.
Agent file access: unknown (outside MLCA).

Workflow: regression 0.1.0
Measurement arm: worker
Task: {'type': 'regression', 'primary_metric': 'rmse', 'metric_reason': 'Use workflow default metric for regression; cnt is a count but no explicit override is requested.', 'targets': ['cnt'], 'time_columns': ['dteday', 'hr'], 'autonomous': True}
Resolved columns: {'target': 'cnt'}
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
{'method': 'time', 'train_fraction': 0.6, 'val_fraction': 0.2, 'seed': 42, 'group_column': '', 'time_column': '', 'time_columns': ['dteday', 'hr']}
Split manifest SHA256: 3e15b48d8f1f500dfe6e6e639def41ab0381a586fb4fc98c69b11c7030b9455f

## 01_onboard_confirm_brief_93XY0Y1S
Follow the recorded task and saved model contract; all fitting uses training rows only.
Parameters: {}

## 02_data_audit_schema_BTC9E8M0
Follow the recorded task and saved model contract; all fitting uses training rows only.
Parameters: {}

## 03_eda_describe_distributions_54FDPF8J
Follow the recorded task and saved model contract; all fitting uses training rows only.
Parameters: {}

## 04_split_freeze_time_0YAATZK4
Follow the recorded task and saved model contract; all fitting uses training rows only.
Parameters: {'method': 'time', 'train_fraction': 0.6, 'val_fraction': 0.2, 'seed': 42, 'group_column': '', 'time_column': ''}

## 05_preprocess_fit_preprocessing_VM7ERF98
Follow the recorded task and saved model contract; all fitting uses training rows only.
Parameters: {}

## 06_features_retain_features_RGQ2HW1B
Follow the recorded task and saved model contract; all fitting uses training rows only.
Parameters: {}

## 07_model_train_baseline_MY9YJS0R
Follow the recorded task and saved model contract; all fitting uses training rows only.
Parameters: {}

## 08_model_train_alternative_VYCC8X2X
Follow the recorded task and saved model contract; all fitting uses training rows only.
Parameters: {}

## 09_validate_score_validation_6CVGV8P9
Score candidates on validation rows: Score the frozen candidates using the task-specific declared primary metric and its direction; supplementary metrics do not replace it. Fit only on training rows, select on validation, and evaluate the frozen choice on test.
Parameters: {}

## 10_test_score_test_RJFPX649
Score the frozen model on test rows: Score the frozen candidates using the task-specific declared primary metric and its direction; supplementary metrics do not replace it. Fit only on training rows, select on validation, and evaluate the frozen choice on test.
Parameters: {}

## 11_interpret_explain_model_CT8GE6DE
Follow the recorded task and saved model contract; all fitting uses training rows only.
Parameters: {}

