import mlca_runtime, json, pickle, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

_protocol = json.loads(inputs['_protocol'].read_text())
task = _protocol['task']
targets = _protocol['targets']
target = targets[0]
primary_metric = task['primary_metric']

val_manifest = json.loads(inputs['09_validate_score_validation_Z73G6FNP/manifest.json'].read_text())

m07 = pickle.loads(inputs['07_model_train_logistic_X39FZFWK/model.pkl'].read_bytes())
m08 = pickle.loads(inputs['08_model_train_alternative_XM6RHY82/alternative_model.pkl'].read_bytes())

val = pd.read_parquet(inputs['val'])
yval = val[target].values

def val_score(model):
    proba = model.predict_proba(val[list(model.feature_names_in_)])
    cls = list(model.classes_)
    pos = task.get('positive_label', cls[1])
    if pos not in cls:
        pos = cls[1]
    yb = (yval == pos).astype(int)
    return roc_auc_score(yb, proba[:, cls.index(pos)])

auc07 = val_score(m07)
auc08 = val_score(m08)
model = m08 if auc08 >= auc07 else m07

def find_threshold(m):
    stack = [m]
    while stack:
        d = stack.pop()
        if isinstance(d, dict):
            for k, v in d.items():
                if 'thresh' in k.lower() and isinstance(v, (int, float)) and not isinstance(v, bool):
                    return float(v)
                stack.append(v)
        elif isinstance(d, list):
            stack.extend(d)
    return None

threshold = find_threshold(val_manifest)
if threshold is None:
    try:
        vm = pd.read_parquet(inputs['09_validate_score_validation_Z73G6FNP/validation_metrics.parquet'])
        for c in vm.columns:
            if c.lower() in ('metric', 'name') and 'value' in vm.columns:
                sub = vm[vm[c].astype(str).str.contains('thresh', case=False)]
                if len(sub):
                    threshold = float(sub['value'].iloc[0]); break
    except Exception:
        pass
if threshold is None:
    threshold = 0.5

test = pd.read_parquet(inputs['test'])
proba = model.predict_proba(test[list(model.feature_names_in_)])
classes = list(model.classes_)
pos_class = task.get('positive_label', classes[1])
if pos_class not in classes:
    pos_class = classes[1]
pos_idx = classes.index(pos_class)
other_class = classes[0] if pos_idx == 1 else classes[1]
ytest = test[target].values
y_bin = (ytest == pos_class).astype(int)
p_pos = proba[:, pos_idx]

def metrics_at(idx):
    yb = y_bin[idx]; pp = p_pos[idx]
    ypred = (pp >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(yb, ypred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    return {
        'roc_auc': roc_auc_score(yb, pp) if len(np.unique(yb)) > 1 else np.nan,
        'pr_auc': average_precision_score(yb, pp),
        'balanced_accuracy': (sens + spec) / 2.0,
        'brier_score': float(np.mean((pp - yb) ** 2)),
        'sensitivity': float(sens),
        'specificity': float(spec),
    }

full = metrics_at(np.arange(len(ytest)))
rng = np.random.RandomState(42)
n = len(ytest)
boot = {k: [] for k in full}
for _ in range(1000):
    idx = rng.randint(0, n, n)
    try:
        mv = metrics_at(idx)
    except Exception:
        continue
    for k, v in mv.items():
        if v is not None and np.isfinite(v):
            boot[k].append(v)

rows = []
for k, v in full.items():
    arr = np.array(boot[k])
    lo, hi = (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))) if len(arr) else (np.nan, np.nan)
    rows.append({'metric': k,
                 'value': round(float(v), 6) if np.isfinite(v) else np.nan,
                 'ci_low': round(lo, 6) if np.isfinite(lo) else np.nan,
                 'ci_high': round(hi, 6) if np.isfinite(hi) else np.nan})
mdf = pd.DataFrame(rows)
mlca_runtime.save_table(mdf, 'test_metrics.csv', caption='Test metrics with 95% bootstrap CI')

pr = mdf[mdf['metric'] == primary_metric].iloc[0]
mlca_runtime.record_metric(primary_metric, float(pr['value']), 'test',
                           ci=[float(pr['ci_low']), float(pr['ci_high'])])
if primary_metric == 'roc_auc':
    mlca_runtime.record_fact_candidate('test_roc_auc', float(pr['value']), source='test_metrics.csv')

ypred_bin = (p_pos >= threshold).astype(int)
ypred_labels = np.where(ypred_bin == 1, pos_class, other_class)
pred_df = pd.DataFrame({
    'row_id': test.index,
    'partition': 'test',
    f'y_true__{target}': ytest,
    f'y_pred__{target}': ypred_labels,
})
for j, c in enumerate(classes):
    pred_df[f'proba__{c}'] = proba[:, j]
mlca_runtime.save_row_outputs(pred_df, 'test_predictions.parquet')

print(f"val_auc07={auc07:.4f} val_auc08={auc08:.4f}")
print(f"test_n={n} threshold={threshold} positive_label={pos_class}")
print(f"test {primary_metric}={pr['value']:.4f} CI=[{pr['ci_low']:.4f}, {pr['ci_high']:.4f}]")
