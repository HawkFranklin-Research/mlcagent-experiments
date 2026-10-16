import mlca_runtime
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())

required = ['targets', 'excluded_columns', 'group_column', 'task']
for k in required:
    if k not in protocol:
        raise KeyError(f"Missing key '{k}' in _protocol. Available keys: {list(protocol.keys())}")

targets = protocol['targets']
excluded = protocol['excluded_columns']
group_col = protocol['group_column']
task = protocol['task']

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

drop = set(targets) | set(excluded)
if group_col is not None:
    drop.add(group_col)

def is_key_like(s):
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return s.nunique(dropna=False) == len(s)
    if pd.api.types.is_integer_dtype(s):
        return s.nunique(dropna=False) == len(s)
    return False

feature_cols = [c for c in train.columns if c not in drop and not is_key_like(train[c])]

preprocessor = pickle.load(open(inputs['05_preprocess_fit_preprocessing_XHDQ4AZ4/preprocessor.pkl'], 'rb'))
selector = pickle.load(open(inputs['06_features_select_features_ZYCM1WYT/selector.pkl'], 'rb'))
model = pickle.load(open(inputs['07_model_train_logistic_D8PWVNVS/model.pkl'], 'rb'))
alt_model = pickle.load(open(inputs['08_model_train_alternative_V3H8V8QN/alternative_model.pkl'], 'rb'))
latest = pickle.load(open(inputs['latest_model'], 'rb'))

def get_input_cols(obj):
    if hasattr(obj, 'feature_names_in_'):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline) and obj.steps:
        first = obj.steps[0][1]
        if hasattr(first, 'feature_names_in_'):
            return list(first.feature_names_in_)
    return None

objects = {
    'preprocessor': preprocessor,
    'selector': selector,
    'model': model,
    'alt_model': alt_model,
    'latest': latest
}

best_pipeline = None
best_steps = -1
for name, obj in objects.items():
    if isinstance(obj, Pipeline):
        cols = get_input_cols(obj)
        if cols is not None and set(cols) == set(feature_cols):
            if len(obj.steps) > best_steps:
                best_steps = len(obj.steps)
                best_pipeline = obj

if best_pipeline is None:
    if isinstance(latest, Pipeline):
        best_pipeline = latest
    else:
        if isinstance(selector, Pipeline):
            best_pipeline = Pipeline([('selector', selector), ('model', latest)])
        elif isinstance(preprocessor, Pipeline):
            best_pipeline = Pipeline([('preprocessor', preprocessor), ('model', latest)])
        else:
            best_pipeline = Pipeline([('preprocessor', preprocessor), ('selector', selector), ('model', latest)])

expected_cols = None
if hasattr(best_pipeline, 'feature_names_in_'):
    expected_cols = list(best_pipeline.feature_names_in_)
elif isinstance(best_pipeline, Pipeline) and best_pipeline.steps:
    first = best_pipeline.steps[0][1]
    if hasattr(first, 'feature_names_in_'):
        expected_cols = list(first.feature_names_in_)

if expected_cols is not None:
    if set(expected_cols) != set(feature_cols):
        raise ValueError("Pipeline expected columns do not match defined feature columns")
    X_val = val[expected_cols].copy()
else:
    X_val = val[feature_cols].copy()

y_val = val[targets[0]]
if isinstance(y_val, pd.DataFrame):
    y_val = y_val.iloc[:, 0]

result = permutation_importance(
    best_pipeline, X_val, y_val,
    scoring='roc_auc',
    n_repeats=20,
    random_state=42
)

df_imp = pd.DataFrame({
    'feature': X_val.columns,
    'importance_mean': result.importances_mean,
    'importance_sd': result.importances_std
})

df_imp = df_imp.sort_values('importance_mean', ascending=False).head(15).reset_index(drop=True)

def sig4(x):
    if pd.isna(x):
        return x
    return float(f"{x:.4g}")

df_imp['importance_mean'] = df_imp['importance_mean'].apply(sig4)
df_imp['importance_sd'] = df_imp['importance_sd'].apply(sig4)

mlca_runtime.save_table(df_imp, "interpretation.csv", caption="Permutation importance on validation rows (ROC AUC drop, 20 repeats)")

top_feature = df_imp.iloc[0]['feature']
mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")

print(f"Validation rows: {len(X_val)}")
print(f"Features evaluated: {len(X_val.columns)}")
print(f"Top feature: {top_feature} (mean importance: {df_imp.iloc[0]['importance_mean']})")
