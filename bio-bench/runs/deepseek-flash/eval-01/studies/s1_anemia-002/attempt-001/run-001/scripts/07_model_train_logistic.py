import mlca_runtime, json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
for key in ('targets', 'excluded_columns', 'task'):
    if key not in protocol:
        raise KeyError(f"Missing {key}; available top-level keys: {sorted(protocol.keys())}")
targets = protocol['targets']
excluded = protocol['excluded_columns']
group_col = protocol.get('group_column')
task = protocol['task']
target = targets[0]
metric_name = task.get('primary_metric', 'roc_auc')

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

with open(inputs['05_preprocess_fit_preprocessing_YDX4CF7Z/preprocessor.pkl'], 'rb') as f:
    preprocessor = pickle.load(f)
with open(inputs['06_features_select_features_FT85WVQ5/selector.pkl'], 'rb') as f:
    selector = pickle.load(f)

def input_cols(obj):
    return list(obj.feature_names_in_) if hasattr(obj, 'feature_names_in_') else None

def out_cols(pre, X):
    try:
        if hasattr(pre, 'get_feature_names_out'):
            return set(pre.get_feature_names_out())
    except Exception:
        pass
    try:
        o = pre.transform(X.head(10))
        if hasattr(o, 'columns'):
            return set(o.columns)
    except Exception:
        pass
    return None

pre_in = input_cols(preprocessor)
sel_in = input_cols(selector)

raw_cols = sel_in or pre_in
if raw_cols is None:
    drop = set(targets) | set(excluded) | ({group_col} if group_col else set())
    raw_cols = []
    for c in train.columns:
        if c in drop:
            continue
        s = train[c]
        n = s.nunique(dropna=True)
        if s.dtype.kind in 'OUS' and n == len(train):
            continue
        if s.dtype.kind in 'iu' and n == len(train) and not set(s.dropna().unique()).issubset({0, 1}):
            continue
        raw_cols.append(c)

X_train = train[raw_cols]
X_val = val[raw_cols]
pre_out = out_cols(preprocessor, X_train)

clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000)

if sel_in is not None and pre_in is not None and set(sel_in) == set(pre_in):
    steps = [('selector', selector), ('clf', clf)]
elif sel_in is not None and pre_out is not None and set(sel_in) == pre_out:
    steps = [('preprocessor', preprocessor), ('selector', selector), ('clf', clf)]
else:
    steps = [('preprocessor', preprocessor), ('selector', selector), ('clf', clf)]

pipeline = Pipeline(steps)
try:
    if pre_out is not None and len(pre_out) <= 300:
        pipeline.set_output(transform='pandas')
except Exception:
    pass

y_train = train[target]
mlca_runtime.fit_record(pipeline, X_train, 'train', y_train)

y_val = val[target]
fitted_clf = pipeline.named_steps['clf']
positive = fitted_clf.classes_[1]
proba_val = pipeline.predict_proba(X_val)[:, 1]
auc_val = roc_auc_score((y_val == positive).astype(int), proba_val)
mlca_runtime.record_metric(metric_name, float(auc_val), 'val')
proba_train = pipeline.predict_proba(X_train)[:, 1]
auc_train = roc_auc_score((y_train == positive).astype(int), proba_train)
mlca_runtime.record_metric(metric_name, float(auc_train), 'train')

mlca_runtime.save_model(pipeline, 'model.pkl')

print(f"train={len(train)} val={len(val)} features={len(raw_cols)}")
print(f"classes={list(fitted_clf.classes_)} positive={positive}")
print(f"train_auc={auc_train:.4f} val_auc={auc_val:.4f}")
