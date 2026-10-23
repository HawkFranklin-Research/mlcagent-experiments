import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
targets = protocol['targets']
excluded = protocol.get('excluded_columns', []) or []
group_col = protocol.get('group_column')

train = pd.read_parquet(inputs['train'])

with open(inputs['05_preprocess_fit_preprocessing_4ZKXP498/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)
with open(inputs['05_preprocess_fit_preprocessing_4ZKXP498/manifest.json'], 'rb') as f:
    manifest = json.load(f)

def is_key_like(s):
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return True
    if pd.api.types.is_integer_dtype(s):
        vals = set(pd.unique(s.dropna()))
        n_notna = int(s.notna().sum())
        if n_notna > 1 and len(vals) == n_notna and not vals.issubset({0, 1}):
            return True
    return False

feature_cols = [c for c in train.columns
                if c not in targets and c not in excluded
                and c != group_col and not is_key_like(train[c])]

target = targets[0]
y = train[target]

if hasattr(pre, 'feature_names_in_'):
    in_cols = list(pre.feature_names_in_)
elif isinstance(pre, Pipeline):
    first = pre.steps[0][1]
    in_cols = list(getattr(first, 'feature_names_in_', feature_cols))
else:
    in_cols = feature_cols

missing = [c for c in in_cols if c not in train.columns]
if missing:
    raise RuntimeError(f"Preprocessor expects missing columns: {missing[:5]}")

k = min(20, len(in_cols))
selector = SelectKBest(mutual_info_classif, k=k)

if isinstance(pre, Pipeline):
    pipe = Pipeline(list(pre.steps) + [('selector', selector)])
else:
    pipe = Pipeline([('preprocessor', pre), ('selector', selector)])

X = train[in_cols]
mlca_runtime.fit_record(pipe, X, 'train', y=y)
mlca_runtime.save_model(pipe, 'selector.pkl')

n_sel = int(pipe.named_steps['selector'].get_support().sum())
mlca_runtime.record_metric('n_input_features', len(in_cols), 'train')
mlca_runtime.record_metric('n_selected_features', n_sel, 'train')
print(f"train rows: {len(train)}, input features: {len(in_cols)}, selected: {n_sel}")
