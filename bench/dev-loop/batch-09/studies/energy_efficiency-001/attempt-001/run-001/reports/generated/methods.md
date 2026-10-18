# Methods draft

Draft generated from the event log. Author review is required.
Agent file access: unknown (outside MLCA).

Workflow: regression 0.1.0
Measurement arm: worker
Task: {'type': 'multioutput_regression', 'primary_metric': 'rmse', 'metric_reason': 'Workflow metric retained; no numeric-outcome override is warranted because heating and cooling loads are equally central and the schema default averaging is sufficient.', 'targets': ['Y1', 'Y2'], 'time_columns': [], 'autonomous': True}
Resolved columns: {'target': 'Y1', 'target_2': 'Y2'}
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
{'method': 'random', 'train_fraction': 0.6, 'val_fraction': 0.2, 'seed': 42, 'group_column': '', 'time_column': ''}
Split manifest SHA256: 31c9502d4784b3aed078649c629f2d2cd7e5f0a0be66d18bf90b96f15a6602c1

## 01_onboard_confirm_brief_HF5HXGDM
Follow the recorded task and saved model contract; all fitting uses training rows only.
Parameters: {}

## 02_data_audit_schema_PZ1VMYV6
Follow the recorded task and saved model contract; all fitting uses training rows only.
Parameters: {}

## 03_eda_describe_distributions_ZB7SBR3E
Follow the recorded task and saved model contract; all fitting uses training rows only.
Parameters: {}

## 04_split_freeze_stratified_J2ZNDSBV
Follow the recorded task and saved model contract; all fitting uses training rows only.
Parameters: {'method': 'random', 'train_fraction': 0.6, 'val_fraction': 0.2, 'seed': 42, 'group_column': '', 'time_column': ''}

## 05_preprocess_fit_preprocessing_0CQ66EDA
Follow the recorded task and saved model contract; all fitting uses training rows only.
Parameters: {}

## 06_features_retain_features_JTARV5X3
Follow the recorded task and saved model contract; all fitting uses training rows only.
Parameters: {}

## 07_model_train_baseline_Q7Z8BHZ9
Follow the recorded task and saved model contract; all fitting uses training rows only.
Parameters: {}

## 08_model_train_alternative_5G4S54TV
Follow the recorded task and saved model contract; all fitting uses training rows only.
Parameters: {}

## 09_validate_score_validation_N590X4VJ
Score candidates on validation rows: Score the frozen candidates using the task-specific declared primary metric and its direction; supplementary metrics do not replace it. Fit only on training rows, select on validation, and evaluate the frozen choice on test.
Parameters: {}

## 10_test_score_test_RT0CSEK4
Score the frozen model on test rows: Score the frozen candidates using the task-specific declared primary metric and its direction; supplementary metrics do not replace it. Fit only on training rows, select on validation, and evaluate the frozen choice on test.
Parameters: {}

## 11_interpret_explain_model_8W4H0Q2F
Follow the recorded task and saved model contract; all fitting uses training rows only.
Parameters: {}

