import json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

proto = json.loads(inputs['_protocol'].read_text())
targets = proto['targets']
if isinstance(targets, str):
    targets = [targets]
excluded = list(proto['excluded_columns'])
group = proto.get('group_column')

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

def is_keylike(s):
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        return len(s) > 0 and s.nunique() == len(s)
    return False

feat_cols = [c for c in train.columns
             if c not in targets and c not in excluded and c != group
             and not is_keylike(train[c])
             and not pd.api.types.is_datetime64_any_dtype(train[c])]
X_train = train[feat_cols].copy()
X_val = val[feat_cols].copy()
y_train = train[targets[0]].values
y_val = val[targets[0]].values

with open(inputs['05_preprocess_fit_preprocessing_CE7RFVQ7/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)
with open(inputs['06_features_select_features_JYFK2080/selector.pkl'], 'rb') as f:
    sel = pickle.load(f)

sel_inputs = list(sel.feature_names_in_) if hasattr(sel, 'feature_names_in_') else None
hgb = HistGradientBoostingClassifier(random_state=42)
if sel_inputs is not None and set(sel_inputs) == set(feat_cols):
    pipe = Pipeline([('selector', sel), ('model', hgb)])
else:
    pipe = Pipeline([('preprocessor', pre), ('selector', sel), ('model', hgb)])

pipe = mlca_runtime.fit_record(pipe, X_train, 'train', y_train)
mlca_runtime.save_model(pipe, 'alternative_model.pkl')

model = pipe.named_steps['model']
classes = list(model.classes_)
proba = pipe.predict_proba(X_val)
present = [c for c in classes if (y_val == c).any()]
if len(present) < len(classes):
    aucs = [roc_auc_score((y_val == c).astype(int), proba[:, classes.index(c)]) for c in present]
    score = float(np.mean(aucs))
else:
    score = float(roc_auc_score(y_val, proba, multi_class='ovr', average='macro', labels=classes))
mlca_runtime.record_metric('roc_auc_ovr', score, 'val')
print(f"train rows: {len(X_train)}, features: {len(feat_cols)}")
print(f"classes: {classes}")
print(f"val roc_auc_ovr: {score:.4f}")
