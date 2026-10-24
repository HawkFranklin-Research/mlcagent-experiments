import mlca_runtime
import json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(Path(inputs['_protocol']).read_text())
train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])
target = protocol['targets'][0]
drop = set(protocol['targets']) | set(protocol['excluded_columns'])
if protocol.get('group_column'):
    drop.add(protocol['group_column'])
raw_cols = [c for c in train.columns if c not in drop]

with open(inputs['05_preprocess_fit_preprocessing_AP3P14P9/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)
with open(inputs['06_features_select_features_NBH1R9WT/selector.pkl'], 'rb') as f:
    sel = pickle.load(f)

sel_in = [str(c) for c in getattr(sel, 'feature_names_in_', [])]
pre_in = [str(c) for c in getattr(pre, 'feature_names_in_', [])]

clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000)
if sel_in and sel_in == list(raw_cols):
    feat = raw_cols
    model = Pipeline([('frontend', sel), ('clf', clf)])
else:
    feat = pre_in if pre_in else raw_cols
    model = Pipeline([('pre', pre), ('sel', sel), ('clf', clf)])

X, y = train[feat], train[target]
mlca_runtime.fit_record(model, X, 'train', y=y)
mlca_runtime.save_model(model, 'model.pkl')

X_val, y_val = val[feat], val[target]
proba = model.predict_proba(X_val)
classes = list(model.classes_)
present = [c for c in classes if (y_val == c).any()]
aucs = [roc_auc_score((y_val == c).astype(int).values, proba[:, classes.index(c)]) for c in present]
val_auc = float(np.mean(aucs))
metric_name = 'roc_auc_ovr_macro' if protocol['task']['primary_metric'] == 'roc_auc_ovr' else protocol['task']['primary_metric']
mlca_runtime.record_metric(metric_name, val_auc, 'val')
print(f"train={train.shape} val={val.shape} features={len(feat)}")
print(f"val_{metric_name}={val_auc:.4f} classes_present={len(present)}/{len(classes)}")
