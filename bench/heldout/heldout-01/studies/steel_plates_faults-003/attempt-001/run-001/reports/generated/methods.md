# Methods draft

Draft generated from the event log. Author review is required.
Agent file access: unknown (outside MLCA).

Workflow: multilabel_classification 0.1.0
Measurement arm: worker
Task: {'type': 'multilabel', 'primary_metric': 'macro_f1', 'metric_reason': '', 'targets': ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults'], 'time_columns': [], 'autonomous': True}
Resolved columns: {'target': 'Pastry', 'target_2': 'Z_Scratch', 'target_3': 'K_Scatch', 'target_4': 'Stains', 'target_5': 'Dirtiness', 'target_6': 'Bumps', 'target_7': 'Other_Faults'}
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
Split manifest SHA256: 058c2a11df524bf3805bf25112ce1eb150347cf5a0a5b477fc9ec48a19cc7b6a

## 01_onboard_confirm_brief_KT5PJNX3
Follow the recorded task and saved model contract; all fitting uses training rows only.
Parameters: {}

## 02_data_count_target_G2CKKZ57
Follow the recorded task and saved model contract; all fitting uses training rows only.
Parameters: {}

## 03_data_audit_schema_ABNT4TQ2
Follow the recorded task and saved model contract; all fitting uses training rows only.
Parameters: {}

## 04_eda_describe_distributions_Y3CSQ6YZ
Follow the recorded task and saved model contract; all fitting uses training rows only.
Parameters: {}

## 05_split_freeze_stratified_B3G71FTX
Follow the recorded task and saved model contract; all fitting uses training rows only.
Parameters: {'method': 'random', 'train_fraction': 0.6, 'val_fraction': 0.2, 'seed': 42, 'group_column': '', 'time_column': ''}

## 06_preprocess_fit_preprocessing_HH50E3RG
Follow the recorded task and saved model contract; all fitting uses training rows only.
Parameters: {}

## 07_features_retain_features_Y0FEJMPH
Follow the recorded task and saved model contract; all fitting uses training rows only.
Parameters: {}

## 08_model_train_baseline_6RRDC1E6
Follow the recorded task and saved model contract; all fitting uses training rows only.
Parameters: {}

## 09_model_train_alternative_D4ZR7T7K
Follow the recorded task and saved model contract; all fitting uses training rows only.
Parameters: {}

## 10_validate_score_validation_E0F8K8XX
Score candidates on validation rows: Score the frozen candidates using the task-specific declared primary metric and its direction; supplementary metrics do not replace it. Fit only on training rows, select on validation, and evaluate the frozen choice on test.
Parameters: {}

## 11_test_score_test_7TGFWKR3
Score the frozen model on test rows: Score the frozen candidates using the task-specific declared primary metric and its direction; supplementary metrics do not replace it. Fit only on training rows, select on validation, and evaluate the frozen choice on test.
Parameters: {}

## 12_interpret_explain_model_8PDW62NP
Follow the recorded task and saved model contract; all fitting uses training rows only.
Parameters: {}

