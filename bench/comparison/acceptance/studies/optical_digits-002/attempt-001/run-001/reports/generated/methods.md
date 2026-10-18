# Methods draft

Draft generated from the event log. Author review is required.
Agent file access: unknown (outside MLCA).

Workflow: representation 0.1.0
Measurement arm: worker
Task: {'type': 'representation', 'primary_metric': 'reconstruction_error', 'metric_reason': 'Representation task uses the workflow default metric.', 'targets': [], 'time_columns': [], 'autonomous': True}
Resolved columns: {}
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
Split manifest SHA256: 1dc37c32840253e8715246d2c17d7853f6485a222257b1170ba7510c772fb8a9

## 01_onboard_confirm_brief_E1EWC1X0
Record brief and data inventory: record the brief answers, resolved columns and file inventory.
Parameters: {}

## 02_data_audit_schema_TDRKJYZK
Audit columns, types and missingness: one row per column: name, dtype, n_unique, n_missing, pct_missing.
Parameters: {}

## 03_eda_describe_distributions_JF3VV49A
Summarise distributions: numeric columns: mean, sd, min, median, max, n_missing; categorical: n_levels, top-level share. Summarize features without class labels.
Parameters: {}

## 04_split_freeze_stratified_AG327PDJ
Random split (60/20/20, seed 42): engine random split; use group or time ordering when the recorded design requires it.
Parameters: {'method': 'random', 'train_fraction': 0.6, 'val_fraction': 0.2, 'seed': 42, 'group_column': '', 'time_column': ''}

## 05_preprocess_fit_preprocessing_XD81KYMB
Fit imputation, encoding and scaling: fit a ColumnTransformer using only training feature columns, excluding every target and identifier named in _protocol. Use median imputation, numeric scaling and categorical encoding as needed.
Parameters: {}

## 06_features_retain_features_75C8096C
Keep all features (no selection): fit a passthrough selector with fit_record on the full frozen training index; preserve feature names and order.
Parameters: {}

## 07_model_train_baseline_93EYVTD0
Fit baseline: fit a complete prediction/reconstruction pipeline with PCA(n_components=min(8, number_of_features)) via fit_record on the original training DataFrame; no supervised outcome is used. Reuse the declared feature contract and exclude all targets.
Parameters: {}

## 08_model_train_alternative_NZ42FXM6
Fit alternative: fit a complete prediction/reconstruction pipeline with MLPRegressor(hidden_layer_sizes=(16,8,16), max_iter=300, random_state=42) via fit_record on the original training DataFrame; no supervised outcome is used. Reuse the declared feature contract and exclude all targets.
Parameters: {}

## 09_validate_score_validation_TA0EMQEK
Score candidates on validation rows: Score the frozen candidates using the task-specific declared primary metric and its direction; supplementary metrics do not replace it. Fit only on training rows, select on validation, and evaluate the frozen choice on test.
Parameters: {}

## 10_test_score_test_S0QXC3C8
Score the frozen model on test rows: Score the frozen candidates using the task-specific declared primary metric and its direction; supplementary metrics do not replace it. Fit only on training rows, select on validation, and evaluate the frozen choice on test.
Parameters: {}

## 11_interpret_explain_model_3G9W1V64
Rank features of the frozen model: describe the frozen model, its selected hyperparameters, inputs, validation evidence and limitations, without fitting.
Parameters: {}

