import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, roc_curve, average_precision_score,
                             balanced_accuracy_score, brier_score_loss, f1_score)

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
task = protocol["task"]
primary_metric = task["primary_metric"]
targets = protocol["targets"]
excluded = list(protocol.get("excluded_columns", []))
group_col = protocol.get("group_column")
target = targets[0]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
y_val = val[target].copy()

drop = set(targets) | set(excluded) | ({group_col} if group_col else set())
cand = [c for c in train.columns if c not in drop]
n = len(train)
key_like = []
for c in cand:
    s = train[c]
    try:
        if s.nunique(dropna=False) == n and not s.isna().any() and (
            s.dtype == object or pd.api.types.is_string_dtype(s) or pd.api.types.is_integer_dtype(s)):
            key_like.append(c)
    except Exception:
        pass
feature_cols = [c for c in cand if c not in key_like]

def lp(key):
    p = inputs.get(key)
    if p is None: return None
    with open(p, 'rb') as f: return pickle.load(f)

prep = lp("05_preprocess_fit_preprocessing_KGJNVG5D/preprocessor.pkl")
sel = lp("06_features_select_features_20NTEHHR/selector.pkl")
candidates = [
    ("logistic", lp("07_model_train_logistic_VP57BH50/model.pkl")),
    ("alternative", lp("08_model_train_alternative_RH6N3E81/alternative_model.pkl")),
]

def has_name(m, keys):
    return isinstance(m, Pipeline) and any(any(k in nme.lower() for k in keys) for nme in m.named_steps)

def assemble(m):
    if isinstance(m, Pipeline):
        first = m.named_steps[list(m.named_steps)[0]]
        fi = getattr(first, 'feature_names_in_', None)
        if fi is not None and set(fi) == set(feature_cols):
            return m
    steps = []
    if prep is not None and not has_name(m, ['prep', 'trans']):
        steps.append(('preprocessor', prep))
    if sel is not None and not has_name(m, ['sel', 'select']):
        steps.append(('selector', sel))
    if isinstance(m, Pipeline):
        steps.extend(list(m.named_steps.items()))
    else:
        steps.append(('model', m))
    return Pipeline(steps)

def align(pipe, X):
    first = pipe.named_steps[list(pipe.named_steps)[0]] if isinstance(pipe, Pipeline) else pipe
    fi = getattr(first, 'feature_names_in_', None)
    if fi is not None and all(c in X.columns for c in fi):
        return X[list(fi)]
    return X[feature_cols]

rows = []
vals = {}
for name, m in candidates:
    pipe = assemble(m)
    Xv = align(pipe, val)
    proba = pipe.predict_proba(Xv)
    est = pipe.named_steps[list(pipe.named_steps)[-1]] if isinstance(pipe, Pipeline) else pipe
    classes = list(est.classes_)
    if len(classes) == 2:
        pos = 1
        if task.get('positive_label') is not None:
            pos = classes.index(task['positive_label'])
        p1 = proba[:, pos]
        y1 = (y_val == classes[pos]).astype(int)
        auc = float(roc_auc_score(y1, p1))
        pr = float(average_precision_score(y1, p1))
        fpr, tpr, ths = roc_curve(y1, p1)
        th = float(ths[int(np.argmax(tpr - fpr))])
        ba = float(balanced_accuracy_score(y1, (p1 >= th).astype(int)))
        brier = float(brier_score_loss(y1, p1))
        rows += [
            dict(model=name, metric='roc_auc', value=round(auc, 6), threshold=round(th, 6)),
            dict(model=name, metric='pr_auc', value=round(pr, 6), threshold=round(th, 6)),
            dict(model=name, metric='balanced_accuracy', value=round(ba, 6), threshold=round(th, 6)),
            dict(model=name, metric='brier', value=round(brier, 6), threshold=round(th, 6)),
        ]
    else:
        auc = float(roc_auc_score(y_val, proba, multi_class='ovr', average='macro', labels=classes))
        yhat = np.array(classes)[np.argmax(proba, axis=1)]
        mf1 = float(f1_score(y_val, yhat, average='macro'))
        rows += [
            dict(model=name, metric='roc_auc_ovr', value=round(auc, 6), threshold=None),
            dict(model=name, metric='f1_macro', value=round(mf1, 6), threshold=None),
        ]
    vals[name] = auc
    mlca_runtime.record_metric(primary_metric, auc, split='val')

df = pd.DataFrame(rows)
mlca_runtime.save_table(df, 'validation_metrics.csv', caption='Validation metrics by candidate model')

higher = primary_metric in {'roc_auc', 'roc_auc_ovr', 'average_precision', 'pr_auc',
                            'accuracy', 'balanced_accuracy', 'f1', 'f1_macro', 'r2'}
best = (max if higher else min)(vals, key=vals.get)
mlca_runtime.record_fact_candidate('best_model', best, source='validation_metrics.csv')
mlca_runtime.record_fact_candidate('best_val_roc_auc', round(vals[best], 6), source='validation_metrics.csv')

print(f"features={len(feature_cols)}")
print(f"best={best} val_{primary_metric}={vals[best]:.4f}")
