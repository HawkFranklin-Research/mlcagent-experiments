import mlca_runtime
import pandas as pd
import json
import pickle
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())

for key in ['targets', 'excluded_columns', 'group_column', 'task']:
    if key not in protocol:
        raise KeyError(f"Missing key '{key}' in _protocol. Available keys: {list(protocol.keys())}")

targets = protocol['targets']
excluded = protocol['excluded_columns']
group_col = protocol['group_column']
task = protocol['task']

train = pd.read_parquet(inputs['train'])
with open(inputs['05_preprocess_fit_preprocessing_CZ0KAB5J/preprocessor.pkl'], 'rb') as f:
    preprocessor = pickle.load(f)

if hasattr(preprocessor, 'feature_names_in_'):
    feature_cols = preprocessor.feature_names_in_
elif isinstance(preprocessor, Pipeline):
    feature_cols = None
    for _, step in preprocessor.steps:
        if hasattr(step, 'feature_names_in_'):
            feature_cols = step.feature_names_in_
            break
    if feature_cols is None:
        all_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group_col]
        feature_cols = []
        for c in all_cols:
            s = train[c]
            if s.dtype == 'object' or s.dtype == 'string':
                if s.nunique() == len(s):
                    continue
            elif pd.api.types.is_integer_dtype(s):
                if s.nunique() == len(s):
                    ss = s.sort_values()
                    if (ss.diff().dropna() == 1).all():
                        continue
            feature_cols.append(c)
else:
    all_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group_col]
    feature_cols = []
    for c in all_cols:
        s = train[c]
        if s.dtype == 'object' or s.dtype == 'string':
            if s.nunique() == len(s):
                continue
        elif pd.api.types.is_integer_dtype(s):
            if s.nunique() == len(s):
                ss = s.sort_values()
                if (ss.diff().dropna() == 1).all():
                    continue
        feature_cols.append(c)

try:
    n_features = len(preprocessor.get_feature_names_out())
except Exception:
    X_sample = train[feature_cols].head(5)
    n_features = preprocessor.transform(X_sample).shape[1]

k = min(20, n_features)
selector = SelectKBest(score_func=mutual_info_classif, k=k)

if isinstance(preprocessor, Pipeline):
    step_name = 'selector'
    existing = [name for name, _ in preprocessor.steps]
    while step_name in existing:
        step_name += '_'
    full_pipeline = Pipeline(list(preprocessor.steps) + [(step_name, selector)])
else:
    full_pipeline = Pipeline([('preprocessor', preprocessor), ('selector', selector)])

X_train = train[feature_cols]
y_train = train[targets[0]]

mlca_runtime.fit_record(full_pipeline, X_train, 'train', y=y_train)
mlca_runtime.save_model(full_pipeline, 'selector.pkl')

print(f"Train shape: {train.shape}")
print(f"Feature columns: {len(feature_cols)}, n_features after preprocessing: {n_features}, k: {k}")
print("Saved selector.pkl")
