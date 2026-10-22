import mlca_runtime
import json
import pickle
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
train = pd.read_parquet(inputs['train'])

with open(inputs['05_preprocess_fit_preprocessing_49B478YE/preprocessor.pkl'], 'rb') as f:
    preprocessor = pickle.load(f)
with open(inputs['06_features_select_features_B4QNDNPN/selector.pkl'], 'rb') as f:
    selector = pickle.load(f)
with open(inputs['07_model_train_cox_5FF75D4Q/model.pkl'], 'rb') as f:
    old_model = pickle.load(f)

targets = protocol.get('targets', [])
if not targets:
    raise ValueError("No targets in protocol")
duration_col = targets[0]
event_col = "Overall Survival Status"
if hasattr(old_model, 'duration_col'):
    duration_col = old_model.duration_col
if hasattr(old_model, 'event_col'):
    event_col = old_model.event_col

if hasattr(preprocessor, 'feature_names_in_'):
    feature_cols = list(preprocessor.feature_names_in_)
elif hasattr(selector, 'feature_names_in_'):
    feature_cols = list(selector.feature_names_in_)
else:
    excluded = set(protocol.get('excluded_columns', []))
    group = protocol.get('group_column', None)
    if group:
        excluded.add(group)
    feature_cols = [c for c in train.columns if c not in targets and c not in excluded]
feature_cols = [c for c in feature_cols if c not in (duration_col, event_col)]

use_preprocessor = True
if isinstance(selector, Pipeline):
    for _, step in selector.named_steps.items():
        if step is preprocessor or (
            hasattr(step, 'feature_names_in_') and hasattr(preprocessor, 'feature_names_in_')
            and list(step.feature_names_in_) == list(preprocessor.feature_names_in_)
        ):
            use_preprocessor = False
            break

X_train = train[feature_cols].copy()
if use_preprocessor and hasattr(preprocessor, 'transform'):
    if hasattr(preprocessor, 'set_output'):
        try:
            preprocessor.set_output(transform='pandas')
        except Exception:
            pass
    Xt = preprocessor.transform(X_train)
else:
    Xt = X_train

if hasattr(selector, 'transform'):
    if hasattr(selector, 'set_output'):
        try:
            selector.set_output(transform='pandas')
        except Exception:
            pass
    Xt = selector.transform(Xt)

if not isinstance(Xt, pd.DataFrame):
    Xt = pd.DataFrame(Xt, index=X_train.index)

fit_df = Xt.copy()
fit_df[duration_col] = train[duration_col]
fit_df[event_col] = train[event_col]

cox = CoxPHFitter(penalizer=0.1, l1_ratio=0.5)
mlca_runtime.fit_record(cox, fit_df, 'train', duration_col=duration_col, event_col=event_col)

model_dict = {
    'preprocessor': preprocessor if use_preprocessor else None,
    'selector': selector,
    'cox': cox,
    'duration_col': duration_col,
    'event_col': event_col,
    'feature_columns': feature_cols
}
mlca_runtime.save_model(model_dict, 'alternative_model.pkl')
print(f"Trained CoxPHFitter on {len(fit_df)} rows, {Xt.shape[1]} features")
