import mlca_runtime, json, pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(Path(inputs['_protocol']).read_text())
targets = protocol['targets']
excluded = list(protocol.get('excluded_columns') or [])
group_col = protocol.get('group_column')
task = protocol['task']
metric = task['primary_metric']

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])
y_col = targets[0] if len(targets) == 1 else targets

prev = pickle.load(open(inputs['07_model_train_logistic_0BAYN51T/model.pkl'], 'rb'))

if hasattr(prev, 'feature_names_in_') and prev.feature_names_in_ is not None:
    model_cols = list(prev.feature_names_in_)
else:
    model_cols = [c for c in train.columns
                  if c not in targets and c not in excluded and c != group_col]

X_tr = train[model_cols].copy()
X_val = val[model_cols].copy()
y_tr = train[y_col]
y_val = val[y_col]

if isinstance(prev, Pipeline):
    steps = list(prev.steps)[:-1]
    hgb_params = {'random_state': 42}
    if y_tr.nunique() > 1 and float(y_tr.value_counts(normalize=True).min()) < 0.10:
        hgb_params['class_weight'] = 'balanced'
    pipe = Pipeline(steps + [('model', HistGradientBoostingClassifier(**hgb_params))])
else:
    pre = pickle.load(open(inputs['05_preprocess_fit_preprocessing_M7A7QE51/preprocessor.pkl'], 'rb'))
    sel = pickle.load(open(inputs['06_features_select_features_8STHQFS1/selector.pkl'], 'rb'))
    pipe = Pipeline([('preprocessor', pre), ('selector', sel),
                     ('model', HistGradientBoostingClassifier(random_state=42))])

fitted = mlca_runtime.fit_record(pipe, X_tr, 'train', y_tr)

proba = fitted.predict_proba(X_val)
classes = fitted.classes_ if hasattr(fitted, 'classes_') else fitted.named_steps['model'].classes_

try:
    auc = float(roc_auc_score(y_val, proba, multi_class='ovr', average='macro', labels=classes))
except ValueError:
    present = set(pd.unique(y_val))
    aucs = []
    for j, c in enumerate(classes):
        if c in present:
            aucs.append(float(roc_auc_score((y_val == c).astype(int), proba[:, j])))
    auc = float(np.mean(aucs))

mlca_runtime.record_metric(metric, auc, 'val')
mlca_runtime.save_model(fitted, 'alternative_model.pkl')
print(f"features={len(model_cols)} classes={len(classes)} val_{metric}={auc:.4f}")
