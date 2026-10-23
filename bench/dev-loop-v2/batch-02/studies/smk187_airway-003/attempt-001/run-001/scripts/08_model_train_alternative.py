import json, pickle
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
targets = list(protocol['targets'])
excluded = list(protocol.get('excluded_columns') or [])
group = protocol.get('group_column')
task = protocol['task']
target = targets[0]

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])
pre = pickle.loads(inputs['05_preprocess_fit_preprocessing_M7AXCKT1/preprocessor.pkl'].read_bytes())
sel = pickle.loads(inputs['06_features_select_features_XFCK1C59/selector.pkl'].read_bytes())

def input_names(obj):
    n = getattr(obj, 'feature_names_in_', None)
    if n is not None:
        return list(n)
    if hasattr(obj, 'named_steps'):
        for step in obj.named_steps.values():
            n = getattr(step, 'feature_names_in_', None)
            if n is not None:
                return list(n)
    return None

pre_names = input_names(pre)
all_cols = list(train.columns)
if pre_names is not None:
    feature_cols = [c for c in pre_names if c in train.columns]
else:
    drop = set(targets) | set(excluded) | ({group} if group else set())
    n = len(train)
    feature_cols = []
    for c in all_cols:
        if c in drop:
            continue
        s = train[c]
        if s.dtype == object or str(s.dtype).startswith('string'):
            if s.nunique(dropna=True) == n:
                continue
        elif pd.api.types.is_integer_dtype(s.dtype):
            if s.nunique(dropna=True) == n and s.min() != s.max():
                continue
        feature_cols.append(c)

y = train[target]
if y.isna().any():
    raise ValueError("target has missing values")
classes = sorted(y.unique().tolist())
cw = 'balanced' if y.value_counts(normalize=True).min() < 0.10 else None
clf = HistGradientBoostingClassifier(random_state=42, class_weight=cw) if cw else HistGradientBoostingClassifier(random_state=42)

sel_names = input_names(sel)
if sel_names is not None and set(sel_names) == set(feature_cols):
    pipeline = Pipeline([('sel', sel), ('clf', clf)])
else:
    pipeline = Pipeline([('pre', pre), ('sel', sel), ('clf', clf)])

pipeline = mlca_runtime.fit_record(pipeline, train[feature_cols], 'train', y=y)
mlca_runtime.save_model(pipeline, 'alternative_model.pkl')

try:
    pos = pipeline.classes_[1]
    idx = list(pipeline.classes_).index(pos)
    proba = pipeline.predict_proba(val[feature_cols])
    y_true = (val[target].values == pos).astype(int)
    auc = roc_auc_score(y_true, proba[:, idx])
    mlca_runtime.record_metric('roc_auc', float(auc), 'val')
    print(f"val roc_auc={auc:.4f} features={len(feature_cols)} class_weight={cw}")
except Exception as e:
    print(f"val_score_skipped: {type(e).__name__}")
