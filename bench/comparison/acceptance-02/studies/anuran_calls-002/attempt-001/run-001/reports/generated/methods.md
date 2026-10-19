# Methods draft

Draft generated from the event log. Author review is required.
Agent file access: unknown (outside MLCA).

Workflow: multilabel_classification 0.1.0
Measurement arm: worker
Task: {'type': 'multilabel', 'primary_metric': 'macro_f1', 'metric_reason': 'Workflow default metric retained for a multilabel acoustic-classification task; no numeric-outcome override applies.', 'targets': ['Family', 'Genus', 'Species'], 'time_columns': [], 'autonomous': True}
Resolved columns: {'target': 'Family', 'target_2': 'Genus', 'target_3': 'Species'}
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

## 01_onboard_confirm_brief_X2DBW9RX
Follow the recorded task and saved model contract; all fitting uses training rows only.
Parameters: {}

## 02_data_audit_schema_6ZXBHMTM
Follow the recorded task and saved model contract; all fitting uses training rows only.
Parameters: {}

## 03_eda_describe_distributions_CW383ZDS
Follow the recorded task and saved model contract; all fitting uses training rows only.
Parameters: {}

## 04_split_freeze_group_M42AXC4V
Follow the recorded task and saved model contract; all fitting uses training rows only.
Parameters: {'method': 'group', 'train_fraction': 0.6, 'val_fraction': 0.2, 'seed': 42, 'group_column': 'RecordID', 'time_column': ''}

## 05_preprocess_fit_preprocessing_A8X1QVXT
Follow the recorded task and saved model contract; all fitting uses training rows only.
Parameters: {}

## 06_features_retain_features_ZGB5PBQC
Follow the recorded task and saved model contract; all fitting uses training rows only.
Parameters: {}

## 07_model_train_baseline_JP4JWWDQ
Follow the recorded task and saved model contract; all fitting uses training rows only.
Parameters: {}

## 08_model_train_alternative_CZ7QSYZK
Follow the recorded task and saved model contract; all fitting uses training rows only.
Parameters: {}

## 09_validate_score_validation_6A3SAZRV
Score candidates on validation rows: Score the frozen candidates using the task-specific declared primary metric and its direction; supplementary metrics do not replace it. Fit only on training rows, select on validation, and evaluate the frozen choice on test.
Parameters: {}

## 10_test_score_test_RFJA80H8
Score the frozen model on test rows: Score the frozen candidates using the task-specific declared primary metric and its direction; supplementary metrics do not replace it. Fit only on training rows, select on validation, and evaluate the frozen choice on test.
Parameters: {}

## 11_interpret_explain_model_A1CDR4JW
Follow the recorded task and saved model contract; all fitting uses training rows only.
Parameters: {}

