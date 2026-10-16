# Methods draft

Draft generated from the event log. Author review is required.
Agent file access: unknown (outside MLCA).

Workflow: regression 0.1.0
Measurement arm: worker
Task: {'type': 'regression', 'primary_metric': 'rmse', 'metric_reason': 'Using workflow default regression metric.', 'targets': ['Rings'], 'time_columns': [], 'autonomous': True}
Resolved columns: {'target': 'Rings'}
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
{'method': 'random', 'train_fraction': 0.6, 'val_fraction': 0.2, 'seed': 42, 'group_column': '', 'time_column': ''}
Split manifest SHA256: 9ed8d81363b05a49f48bcba8f2031e63b27615faedc19925a3cf668fa08dc1e5

## 01_onboard_confirm_brief_QVM56BW3
Follow the recorded task and saved model contract; all fitting uses training rows only.
Parameters: {}

## 02_data_audit_schema_NQ7G12FJ
Follow the recorded task and saved model contract; all fitting uses training rows only.
Parameters: {}

## 03_eda_describe_distributions_182ECTMF
Follow the recorded task and saved model contract; all fitting uses training rows only.
Parameters: {}

## 04_split_freeze_stratified_P52VBDDG
Follow the recorded task and saved model contract; all fitting uses training rows only.
Parameters: {'method': 'random', 'train_fraction': 0.6, 'val_fraction': 0.2, 'seed': 42, 'group_column': '', 'time_column': ''}

## 05_preprocess_fit_preprocessing_JT5JG4F1
Follow the recorded task and saved model contract; all fitting uses training rows only.
Parameters: {}

## 06_features_retain_features_6CDSH3YD
Follow the recorded task and saved model contract; all fitting uses training rows only.
Parameters: {}

## 07_model_train_baseline_F3SAKS84
Follow the recorded task and saved model contract; all fitting uses training rows only.
Parameters: {}

## 08_model_train_alternative_G7VTXE7A
Follow the recorded task and saved model contract; all fitting uses training rows only.
Parameters: {}

## 09_validate_score_validation_WEK9FDVR
Follow the recorded task and saved model contract; all fitting uses training rows only.
Parameters: {}

## 10_test_score_test_9H2B9T9X
Follow the recorded task and saved model contract; all fitting uses training rows only.
Parameters: {}

## 11_interpret_explain_model_YFQTKT18
Follow the recorded task and saved model contract; all fitting uses training rows only.
Parameters: {}

