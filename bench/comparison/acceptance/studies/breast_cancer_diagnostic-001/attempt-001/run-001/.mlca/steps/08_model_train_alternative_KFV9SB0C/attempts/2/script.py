import json, pickle
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

SEED = 42
mlca_runtime.set_seed(SEED)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())

targets = protocol['targets']
excluded = protocol['excluded_columns']
group_col = protocol.get('group_column')
primary_metric = protocol['task']['primary_metric']
target = targets[0]

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)
feature_cols = [c for c in train.columns if c not in drop]
X_train = train[feature_cols]
y_train = train[target]
X_val = val[feature_cols]
y_val = val[target]

with open(inputs['07_model_train_logistic_BMNCC8GS/model.pkl'], 'rb') as f:
    base = pickle.load(f)

if isinstance(base, Pipeline) and base.steps:
    last = base.steps[-1][1]
    if hasattr(last, 'predict') or hasattr(last, 'predict_proba'):
        steps = list(base.steps[:-1])
    else:
        steps = list(base.steps)
    steps = [(n, s) for n, s in steps if n != 'clf']
    steps.append(('clf', HistGradientBoostingClassifier(random_state=SEED)))
    est = Pipeline(steps)
    if hasattr(base, 'feature_names_in_'):
        want = list(base.feature_names_in_)
        if want != feature_cols:
            X_train = train[want]
            X_val = val[want]
else:
    with open(inputs['05_preprocess_fit_preprocessing_WZN8YG48/preprocessor.pkl'], 'rb') as f:
        pre = pickle.load(f)
    with open(inputs['06_features_select_features_6Y9H969T/selector.pkl'], 'rb') as f:
        sel = pickle.load(f)
    est = Pipeline([
        ('preprocessor', pre),
        ('selector', sel),
        ('clf', HistGradientBoostingClassifier(random_state=SEED)),
    ])

vc = y_train.value_counts(normalize=True)
if len(vc) >= 2 and vc.min() < 0.10:
    est.set_params(clf__class_weight='balanced')

est = mlca_runtime.fit_record(est, X_train, 'train', y=y_train)
mlca_runtime.save_model(est, 'alternative_model.pkl')

if primary_metric == 'roc_auc':
    clf = est.named_steps['clf']
    proba = est.predict_proba(X_val)
    classes = clf.classes_
    pos_idx = 1
    y_bin = (y_val == classes[pos_idx]).astype(int)
    auc = roc_auc_score(y_bin, proba[:, pos_idx])
    mlca_runtime.record_metric('roc_auc', float(auc), 'val')
    print(f"val roc_auc={auc:.4f} pos_label={classes[pos_idx]}")

print(f"train={len(X_train)} val={len(X_val)} features={len(X_train.columns)}")
