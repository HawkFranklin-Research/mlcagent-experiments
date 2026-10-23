import mlca_runtime, json, pickle, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
targets = protocol['targets']
excluded = protocol.get('excluded_columns', [])
group_col = protocol.get('group_column')
primary = protocol['task']['primary_metric']
task_type = protocol['task']['type']
target = targets[0]

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])
test = pd.read_parquet(inputs['test'])

def load_model(p):
    with open(p, 'rb') as f:
        return pickle.load(f)

pre_path = inputs['05_preprocess_fit_preprocessing_DXVQHGBE/preprocessor.pkl']
sel_path = inputs['06_features_select_features_9JJFXST3/selector.pkl']
raw_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group_col]
sample = train[raw_cols].head(5)

def is_complete(m):
    try:
        m.predict_proba(sample); return True
    except Exception:
        return False

def get_full(m):
    if is_complete(m): return m
    return Pipeline([('pre', load_model(pre_path)), ('sel', load_model(sel_path)), ('model', m)])

def score_mc(y, proba, classes):
    classes = list(classes); present = [c for c in np.unique(y) if c in classes]
    if set(present) == set(classes):
        try:
            return float(roc_auc_score(y, proba, multi_class='ovr', average='macro', labels=classes))
        except ValueError:
            pass
    try:
        vals = [roc_auc_score((y == c).astype(int), proba[:, classes.index(c)]) for c in present]
        return float(np.mean(vals)) if vals else float('nan')
    except ValueError:
        return float('nan')

candidates = {
    'logistic': inputs['07_model_train_logistic_6BHE1H1F/model.pkl'],
    'alternative': inputs['08_model_train_alternative_SGR2R7DJ/alternative_model.pkl'],
}
fitted = {}
val_scores = {}
for name, path in candidates.items():
    m = get_full(load_model(path))
    cols = list(m.feature_names_in_) if hasattr(m, 'feature_names_in_') else raw_cols
    val_scores[name] = score_mc(val[target].values, m.predict_proba(val[cols]), m.classes_)
    fitted[name] = (m, cols)

sel = max(val_scores, key=val_scores.get)
model, cols = fitted[sel]
classes = model.classes_
proba = model.predict_proba(test[cols])
y_test = test[target].values
pred = classes[np.argmax(proba, axis=1)]

auc = score_mc(y_test, proba, classes)
f1 = float(f1_score(y_test, pred, average='macro', labels=classes, zero_division=0))

rng = np.random.RandomState(42)
n = len(test)
boot_auc, boot_f1 = [], []
for _ in range(1000):
    idx = rng.randint(0, n, n)
    yb, pb = y_test[idx], proba[idx]
    a = score_mc(yb, pb, classes)
    f = float(f1_score(yb, classes[np.argmax(pb, axis=1)], average='macro', labels=classes, zero_division=0))
    if not np.isnan(a): boot_auc.append(a)
    if not np.isnan(f): boot_f1.append(f)

auc_ci = (float(np.percentile(boot_auc, 2.5)), float(np.percentile(boot_auc, 97.5)))
f1_ci = (float(np.percentile(boot_f1, 2.5)), float(np.percentile(boot_f1, 97.5)))

metrics_df = pd.DataFrame([
    {'metric': primary, 'value': auc, 'ci_low': auc_ci[0], 'ci_high': auc_ci[1]},
    {'metric': 'f1_macro', 'value': f1, 'ci_low': f1_ci[0], 'ci_high': f1_ci[1]},
]).round(4)
mlca_runtime.save_table(metrics_df, 'test_metrics.csv')

pred_df = pd.DataFrame({
    'row_id': test.index,
    'partition': 'test',
    f'y_true__{target}': y_test,
    f'y_pred__{target}': pred,
})
for j, c in enumerate(classes):
    pred_df[f'proba__{c}'] = proba[:, j]
mlca_runtime.save_row_outputs(pred_df, 'test_predictions.parquet')

mlca_runtime.record_metric(primary, auc, 'test', ci=list(auc_ci))
mlca_runtime.record_fact_candidate('test_roc_auc', auc, source='test_metrics.csv')

print(f"selected={sel} val_{primary}={val_scores[sel]:.4f}")
print(f"test_{primary}={auc:.4f} CI=({auc_ci[0]:.4f},{auc_ci[1]:.4f})")
print(f"test_f1_macro={f1:.4f}")
print(f"n_test={n}")
