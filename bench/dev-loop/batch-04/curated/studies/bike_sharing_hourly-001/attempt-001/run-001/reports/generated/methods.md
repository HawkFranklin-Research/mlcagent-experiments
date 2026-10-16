# Methods draft

Draft generated from the event log. Author review is required.
Agent file access: unknown (outside MLCA).

Workflow: regression 0.1.0
Measurement arm: worker
Task: {'type': 'regression', 'primary_metric': 'poisson_deviance', 'metric_reason': 'The target cnt is a nonnegative integer count whose variance tends to scale with its mean, making Poisson deviance an appropriate count-regression metric.', 'targets': ['cnt'], 'time_columns': ['dteday', 'hr'], 'autonomous': True}
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
{'method': 'group', 'train_fraction': 0.6, 'val_fraction': 0.2, 'seed': 42, 'group_column': 'dteday', 'time_column': 'dteday'}
Split manifest SHA256: b865df568f639fa332251fd7c6baea324e011a03a7974f42ba27d9ddf812fafb

## 01_onboard_confirm_brief_MEH0HP65
Follow the recorded task and saved model contract; all fitting uses training rows only.
Parameters: {}

## 02_data_audit_schema_T022FTVG
Follow the recorded task and saved model contract; all fitting uses training rows only.
Parameters: {}

## 03_eda_describe_distributions_EE7JDAAR
Follow the recorded task and saved model contract; all fitting uses training rows only.
Parameters: {}

## 04_split_freeze_time_R95R5929
Follow the recorded task and saved model contract; all fitting uses training rows only.
Parameters: {'method': 'group', 'train_fraction': 0.6, 'val_fraction': 0.2, 'seed': 42, 'group_column': 'dteday', 'time_column': 'dteday'}

## 05_preprocess_fit_preprocessing_DVNTE848
Follow the recorded task and saved model contract; all fitting uses training rows only.
Parameters: {}

## 06_features_retain_features_TBVVBD2H
Follow the recorded task and saved model contract; all fitting uses training rows only.
Parameters: {}

## 07_model_train_baseline_5MMHKFNM
Follow the recorded task and saved model contract; all fitting uses training rows only.
Parameters: {}

## 08_model_train_alternative_SHDNC9V9
Follow the recorded task and saved model contract; all fitting uses training rows only.
Parameters: {}

## 09_validate_score_validation_2WWEBTQD
Score candidates on validation rows: Score the frozen candidates using the task-specific declared primary metric and its direction; supplementary metrics do not replace it. Fit only on training rows, select on validation, and evaluate the frozen choice on test.
Parameters: {}

## 10_test_score_test_R0KNBTN2
Score the frozen model on test rows: Score the frozen candidates using the task-specific declared primary metric and its direction; supplementary metrics do not replace it. Fit only on training rows, select on validation, and evaluate the frozen choice on test.
Parameters: {}

## 11_interpret_explain_model_HSC8ZX1T
Follow the recorded task and saved model contract; all fitting uses training rows only.
Parameters: {}

