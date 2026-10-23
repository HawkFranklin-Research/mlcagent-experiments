import json
import pickle
import pandas as pd
import mlca_runtime
from pathlib import Path
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(Path(inputs['_protocol']).read_text())

targets = protocol['targets']
excluded = protocol.get('excluded_columns', []) or []
group_col = protocol.get('group_column')
target = targets[0]

train = pd.read_parquet(inputs['train'])

with open(inputs['05_preprocess_fit_preprocessing_W6YRPZJC/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)

fni = getattr(pre, 'feature_names_in_', None)
if fni is not None and len(fni) > 0:
    feature_cols = list(fni)
else:
    drop = set(targets) | set(excluded)
    if group_col:
        drop.add(group_col)
    feature_cols = [c for c in train.columns if c not in drop]

X = train[feature_cols]
y = train[target]

pre_is_pipe = isinstance(pre, Pipeline)
pre_has_sel = pre_is_pipe and (
    pre.steps[-1][0] == 'selector' or isinstance(pre.steps[-1][1], SelectKBest)
)

if pre_has_sel:
    saved = pre
    k_val = int(saved.steps[-1][1].k) if isinstance(saved.steps[-1][1], SelectKBest) else -1
else:
    try:
        n_post = int(pre.transform(X.head(3)).shape[1])
    except Exception:
        n_post = len(feature_cols)
    k_val = min(20, n_post)
    saved = Pipeline([
        ('preprocessor', pre),
        ('selector', SelectKBest(mutual_info_classif, k=k_val)),
    ])

mlca_runtime.fit_record(saved, X, 'train', y=y)
mlca_runtime.save_model(saved, 'selector.pkl')

mlca_runtime.record_metric('n_features_in', float(len(feature_cols)), 'train')
mlca_runtime.record_metric('k_selected', float(k_val), 'train')
print(f"n_in={len(feature_cols)} k={k_val} n_rows={len(train)}")
print(f"pre_type={type(pre).__name__} has_selector={pre_has_sel}")
