import mlca_runtime
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

# Load protocol
protocol = json.loads(inputs['_protocol'].read_text())
targets = protocol['targets']
excluded = protocol.get('excluded_columns', [])
group = protocol.get('group_column', None)
task = protocol['task']
primary_metric = task['primary_metric']

# Load data
train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])
print(f"Train shape: {train.shape}")
print(f"Val shape: {val.shape}")

# Define feature columns
feature_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group]
# Exclude key-like columns (object columns with all distinct values)
for c in feature_cols[:]:
    if train[c].dtype == 'object' and train[c].nunique() == len(train):
        feature_cols.remove(c)

# Load candidate pipelines
def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

candidates = []
for key in ['latest_model', '08_model_train_logistic_9FB709K5/model.pkl',
            '09_model_train_alternative_8G8NH389/alternative_model.pkl']:
    try:
        obj = load_pkl(inputs[key])
        if hasattr(obj, 'predict_proba'):
            candidates.append((key, obj))
    except Exception:
        pass

if candidates:
    # pick the one with most Pipeline steps
    best_key, pipe = max(candidates, key=lambda x: len(x[1].steps) if isinstance(x[1], Pipeline) else 0)
else:
    # fallback: build from saved components
    pre = load_pkl(inputs['06_preprocess_fit_preprocessing_2FS9KME6/preprocessor.pkl'])
    sel = load_pkl(inputs['07_features_retain_features_SY1WG08Y/selector_all.pkl'])
    model = load_pkl(inputs['08_model_train_logistic_9FB709K5/model.pkl'])
    pipe = Pipeline([('pre', pre), ('sel', sel), ('model', model)])

# Get expected feature names
def get_feature_names(p):
    if hasattr(p, 'feature_names_in_'):
        return list(p.feature_names_in_)
    if isinstance(p, Pipeline):
        first = p.steps[0][1]
        if hasattr(first, 'feature_names_in_'):
            return list(first.feature_names_in_)
    return None

expected = get_feature_names(pipe)
if expected is not None:
    X_val = val[expected]
else:
    X_val = val[feature_cols]

y_val = val[targets[0]]

# Permutation importance
result = permutation_importance(pipe, X_val, y_val, scoring='roc_auc_ovr',
                                n_repeats=20, random_state=42, n_jobs=1)

imp_mean = result.importances_mean
imp_std = result.importances_std
feat_names = X_val.columns.tolist()

df = pd.DataFrame({
    'feature': feat_names,
    'importance_mean': imp_mean,
    'importance_sd': imp_std
})
df = df.sort_values('importance_mean', ascending=False).head(15)
df['importance_mean'] = df['importance_mean'].apply(lambda x: float(f"{x:.4g}"))
df['importance_sd'] = df['importance_sd'].apply(lambda x: float(f"{x:.4g}"))

mlca_runtime.save_table(df, "interpretation.csv",
                        caption="Permutation importance on validation set (top 15 features)")
top_feature = df.iloc[0]['feature']
mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")

print(f"Top feature: {top_feature} (mean importance: {df.iloc[0]['importance_mean']:.4f})")
