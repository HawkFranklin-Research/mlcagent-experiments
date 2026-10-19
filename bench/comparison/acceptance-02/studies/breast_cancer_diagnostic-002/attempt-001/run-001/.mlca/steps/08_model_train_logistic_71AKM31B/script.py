import mlca_runtime
import json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp['_protocol'].read_text())
targets = protocol['targets']
excluded = list(protocol.get('excluded_columns', []))
group_col = protocol.get('group_column')

train = pd.read_parquet(inp['train'])
val = pd.read_parquet(inp['val'])

with open(inp['06_preprocess_fit_preprocessing_09VH2GD4/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)
with open(inp['07_features_select_features_B35TTT17/selector.pkl'], 'rb') as f:
    sel = pickle.load(f)

pre_in = list(pre.feature_names_in_) if hasattr(pre, 'feature_names_in_') else None
sel_in = list(sel.feature_names_in_) if hasattr(sel, 'feature_names_in_') else None
sel_has_pre = pre_in is not None and sel_in is not None and set(sel_in) == set(pre_in)

clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000)
if sel_has_pre:
    pipeline = Pipeline([('sel', sel), ('clf', clf)])
    raw_cols = sel_in
else:
    pipeline = Pipeline([('prep', pre), ('sel', sel), ('clf', clf)])
    if pre_in is not None:
        raw_cols = pre_in
    else:
        drop = set(targets) | set(excluded) | ({group_col} if group_col else set())
        raw_cols = [c for c in train.columns if c not in drop]

y_train = train[targets[0]]
X_train = train[raw_cols].copy()
y_val = val[targets[0]]
X_val = val[raw_cols].copy()

fitted = mlca_runtime.fit_record(pipeline, X_train, 'train', y=y_train)

proba = fitted.predict_proba(X_val)
classes = list(fitted.classes_)
pos_idx = 1
y_bin = (y_val == classes[pos_idx]).astype(int)
auc = float(roc_auc_score(y_bin, proba[:, pos_idx]))
mlca_runtime.record_metric('roc_auc', auc, 'val')
mlca_runtime.save_model(fitted, 'model.pkl')
print(f"train={X_train.shape} val={X_val.shape} val_auc={auc:.4f} pos_class={classes[pos_idx]}")
