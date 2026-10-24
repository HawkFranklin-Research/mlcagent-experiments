# Methods draft

Draft generated from the event log. Author review is required.
Agent file access: unknown (outside MLCA).

Workflow: regression 0.1.0
Measurement arm: worker
Task: {'type': 'regression', 'targets': ['cnt'], 'primary_metric': 'rmse', 'time_columns': ['dteday', 'hr']}
Resolved columns: {'target': 'cnt'}
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
{'method': 'time', 'train_fraction': 0.6, 'val_fraction': 0.2, 'seed': 42, 'group_column': '', 'time_column': '', 'time_columns': ['dteday', 'hr']}
Split manifest SHA256: 3e15b48d8f1f500dfe6e6e639def41ab0381a586fb4fc98c69b11c7030b9455f

## 01_onboard_confirm_brief_APND16PZ
Follow the recorded task and saved model contract; all fitting uses training rows only.
Parameters: {}

## 02_data_audit_schema_0M5B2SJR
Follow the recorded task and saved model contract; all fitting uses training rows only.
Parameters: {}

## 03_eda_describe_distributions_E2SABQDK
Follow the recorded task and saved model contract; all fitting uses training rows only.
Parameters: {}

## 04_split_freeze_stratified_D7F0TR9Y
Follow the recorded task and saved model contract; all fitting uses training rows only.
Parameters: {'method': 'random', 'train_fraction': 0.6, 'val_fraction': 0.2, 'seed': 42, 'group_column': '', 'time_column': ''}

## 05_preprocess_fit_preprocessing_68CQ5GG8
Follow the recorded task and saved model contract; all fitting uses training rows only.
Parameters: {}

## 06_features_retain_features_T3TQSW0S
Follow the recorded task and saved model contract; all fitting uses training rows only.
Parameters: {}

## 07_model_train_baseline_4G62P28A
Follow the recorded task and saved model contract; all fitting uses training rows only.
Parameters: {}

## 08_validate_score_validation_BRBDG35B
Score candidates on validation rows: Score the frozen candidates using the task-specific declared primary metric and its direction; supplementary metrics do not replace it. Fit only on training rows, select on validation, and evaluate the frozen choice on test.
Parameters: {}

## 09_test_score_test_BV17BQGZ
Score the frozen model on test rows: Score the frozen candidates using the task-specific declared primary metric and its direction; supplementary metrics do not replace it. Fit only on training rows, select on validation, and evaluate the frozen choice on test. Per-row test predictions are saved locally for external paired comparison.
Parameters: {}

## 10_interpret_explain_model_SDR8FH6M
Follow the recorded task and saved model contract; all fitting uses training rows only.
Parameters: {}

