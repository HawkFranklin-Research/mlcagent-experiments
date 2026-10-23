import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              balanced_accuracy_score, brier_score_loss,
                              confusion_matrix, roc_curve, f1_score)

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())
task = proto["task"]; targets = proto["targets"]
excluded = proto.get("excluded_columns", []); group_col = proto.get("group_column")
primary_metric = task["primary_metric"]; task_type = task["type"]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
test = pd.read_parquet(inputs["test"])

# Select the frozen model using validation results only
model_path = inputs["latest_model"]
try:
    vm = pd.read_csv(inputs["10_validate_score_validation_EHS3PS8T/validation_metrics.csv"])
    c = [x.lower() for x in vm.columns]
    if {'model', 'metric', 'value'} <= set(c):
        mc, mc2, vc = vm.columns[c.index('model')], vm.columns[c.index('metric')], vm.columns[c.index('value')]
        sub = vm[vm[mc2] == primary_metric]
        hb = primary_metric in ('roc_auc', 'roc_auc_ovr', 'pr_auc', 'f1_macro', 'balanced_accuracy')
        if len(sub) >= 2:
            best = str(sub.sort_values(vc, ascending=not hb).iloc[0][mc]).lower()
            if '09' in best or 'alt' in best:
                model_path = inputs["09_model_train_alternative_N0C35K8C/alternative_model.pkl"]
            else:
                model_path = inputs["08_model_train_logistic_1JG6Z9GW/model.pkl"]
except Exception:
    pass

with open(model_path, 'rb') as f:
    model = pickle.load(f)

target_set = set(targets) | set(excluded)
if group_col:
    target_set.add(group_col)
feature_cols = [c for c in train.columns if c not in target_set]
use_cols = list(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else feature_cols
use_cols = [c for c in use_cols if c in test.columns]

X_test = test[use_cols]
proba = model.predict_proba(X_test)
classes = list(model.classes_)
y_pred = model.predict(X_test)
X_val = val[use_cols]
proba_val = model.predict_proba(X_val)

if task_type == 'binary':
    y_test_col = test[targets[0]]; y_val_col = val[targets[0]]
    pos_label = classes[1]
    ytb = (y_test_col == pos_label).astype(int).values
    yvb = (y_val_col == pos_label).astype(int).values
    pt = proba[:, 1]; pv = proba_val[:, 1]
    fpr, tpr, thrs = roc_curve(yvb, pv)
    thr = float(thrs[np.argmax(tpr - fpr)])
    ypb = (pt >= thr).astype(int)
    print(f"thr={thr:.4f} pos={pos_label}")

    def _sens(yt, yp):
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        return float(tp / (tp + fn)) if (tp + fn) else np.nan

    def _spec(yt, yp):
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        return float(tn / (tn + fp)) if (tn + fp) else np.nan

    mfns = {
        'roc_auc': lambda yt, yp, pr: roc_auc_score(yt, pr) if len(np.unique(yt)) > 1 else np.nan,
        'pr_auc': lambda yt, yp, pr: average_precision_score(yt, pr) if len(np.unique(yt)) > 1 else np.nan,
        'balanced_accuracy': lambda yt, yp, pr: balanced_accuracy_score(yt, yp),
        'brier': lambda yt, yp, pr: brier_score_loss(yt, pr),
        'sensitivity': lambda yt, yp, pr: _sens(yt, yp),
        'specificity': lambda yt, yp, pr: _spec(yt, yp),
    }

    def boot(fn, yt, yp, pr, n=1000, seed=42):
        rng = np.random.default_rng(seed); vals = []; N = len(yt)
        for _ in range(n):
            idx = rng.integers(0, N, N)
            try:
                v = fn(yt[idx], yp[idx], pr[idx])
                if not (isinstance(v, float) and np.isnan(v)):
                    vals.append(v)
            except Exception:
                pass
        if not vals:
            return np.nan, np.nan, np.nan
        a = np.array(vals)
        return float(a.mean()), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

    rows = []
    for m, fn in mfns.items():
        try:
            point = float(fn(ytb, ypb, pt))
        except Exception:
            point = np.nan
        _, lo, hi = boot(fn, ytb, ypb, pt)
        rows.append({'metric': m, 'value': point, 'ci_low': lo, 'ci_high': hi})
    tm = pd.DataFrame(rows)
    tm[['value', 'ci_low', 'ci_high']] = tm[['value', 'ci_low', 'ci_high']].round(4)
    mlca_runtime.save_table(tm, "test_metrics.csv", caption="Test metrics with 95% bootstrap CI")

    pr_row = tm[tm['metric'] == primary_metric]
    if len(pr_row):
        v = float(pr_row.iloc[0]['value'])
        lo = float(pr_row.iloc[0]['ci_low']); hi = float(pr_row.iloc[0]['ci_high'])
        mlca_runtime.record_metric(primary_metric, v, "test", ci=[lo, hi])
        print(f"{primary_metric} test={v:.4f} ci=[{lo:.4f},{hi:.4f}]")

    if primary_metric == 'roc_auc':
        ra = tm[tm['metric'] == 'roc_auc'].iloc[0]
        mlca_runtime.record_fact_candidate("test_roc_auc", float(ra['value']), source="test_metrics.csv")

    pred = pd.DataFrame({'row_id': test.index, 'partition': 'test'})
    pred[f'y_true__{targets[0]}'] = y_test_col.values
    pred[f'y_pred__{targets[0]}'] = y_pred
    for i, cls in enumerate(classes):
        pred[f'proba__{cls}'] = proba[:, i]
    mlca_runtime.save_row_outputs(pred, "test_predictions.parquet")

elif task_type == 'multiclass':
    y_test_col = test[targets[0]]
    try:
        roc_auc_macro = float(roc_auc_score(y_test_col, proba, labels=classes,
                                            multi_class='ovr', average='macro'))
    except Exception:
        present = [c for c in classes if (y_test_col == c).any()]
        scores = [roc_auc_score((y_test_col == c).astype(int), proba[:, classes.index(c)]) for c in present]
        roc_auc_macro = float(np.mean(scores))
    f1_macro = float(f1_score(y_test_col, y_pred, average='macro'))

    rng = np.random.default_rng(42); vals = []; N = len(y_test_col)
    for _ in range(1000):
        idx = rng.integers(0, N, N)
        try:
            vals.append(f1_score(y_test_col.iloc[idx], y_pred[idx], average='macro'))
        except Exception:
            pass
    a = np.array(vals)
    f1_lo, f1_hi = float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

    tm = pd.DataFrame([
        {'metric': 'roc_auc_ovr', 'value': round(roc_auc_macro, 4), 'ci_low': np.nan, 'ci_high': np.nan},
        {'metric': 'f1_macro', 'value': round(f1_macro, 4), 'ci_low': round(f1_lo, 4), 'ci_high': round(f1_hi, 4)},
    ])
    mlca_runtime.save_table(tm, "test_metrics.csv", caption="Test metrics")
    pr_row = tm[tm['metric'] == primary_metric]
    if len(pr_row):
        v = float(pr_row.iloc[0]['value'])
        lo = pr_row.iloc[0]['ci_low']; hi = pr_row.iloc[0]['ci_high']
        ci = None if pd.isna(lo) else [float(lo), float(hi)]
        mlca_runtime.record_metric(primary_metric, v, "test", ci=ci)

    pred = pd.DataFrame({'row_id': test.index, 'partition': 'test'})
    for t in targets:
        pred[f'y_true__{t}'] = test[t].values
        pred[f'y_pred__{t}'] = y_pred
    for i, cls in enumerate(classes):
        pred[f'proba__{cls}'] = proba[:, i]
    mlca_runtime.save_row_outputs(pred, "test_predictions.parquet")
