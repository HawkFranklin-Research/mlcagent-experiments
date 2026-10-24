import mlca_runtime, json, pickle, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
out = mlca_runtime.output_dir()

train = pd.read_parquet(inp['train'])
val = pd.read_parquet(inp['val'])
protocol = json.loads(inp['_protocol'].read_text())
target = protocol['targets'][0]
n_classes = protocol['task'].get('n_classes', 2)

with open(inp['latest_model'], 'rb') as f:
    model = pickle.load(f)

feature_cols = list(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else [
    c for c in val.columns if c != target and c not in protocol['excluded_columns'] and c != protocol.get('group_column')]
excluded = set(protocol['excluded_columns']) | set(protocol['targets']) | ({protocol.get('group_column')} - {None})
bad = [c for c in feature_cols if c in excluded]
if bad:
    raise ValueError(f"model expects excluded columns: {bad[:5]}")

X_val = val[feature_cols]
y_val = val[target]
n_features = len(feature_cols)
print(f"val_rows={len(val)} input_features={n_features}")

def get_coef():
    est = model
    if hasattr(model, 'named_steps'):
        for name in reversed(list(model.named_steps)):
            s = model.named_steps[name]
            if hasattr(s, 'coef_') or hasattr(s, 'feature_importances_'):
                est = s; break
    if hasattr(est, 'coef_'):
        c = np.abs(est.coef_)
        c = c.sum(axis=0) if c.ndim > 1 and c.shape[0] > 1 else c.ravel()
    elif hasattr(est, 'feature_importances_'):
        c = est.feature_importances_
    else:
        return None
    if len(c) == n_features:
        return c
    if hasattr(model, 'named_steps'):
        support = None
        for name in list(model.named_steps):
            s = model.named_steps[name]
            if s is est: break
            if hasattr(s, 'get_support'): support = s.get_support()
        if support is not None and len(support) == n_features and support.sum() == len(c):
            full = np.zeros(n_features); full[support] = c
            return full
    return None

def score_fn(X):
    if hasattr(model, 'predict_proba'):
        p = model.predict_proba(X)
        return roc_auc_score(y_val, p[:, 1]) if n_classes == 2 else roc_auc_score(y_val, p, multi_class='ovr', average='macro')
    d = model.decision_function(X)
    return roc_auc_score(y_val, d) if n_classes == 2 else roc_auc_score(y_val, d, multi_class='ovr', average='macro')

def permute(feats, n_repeats):
    base = score_fn(X_val)
    rng = np.random.RandomState(42)
    rows = []
    for f in feats:
        drops = []
        for _ in range(n_repeats):
            Xp = X_val.copy()
            Xp[f] = rng.permutation(Xp[f].values)
            drops.append(base - score_fn(Xp))
        rows.append((f, float(np.mean(drops)), float(np.std(drops))))
    return rows

if n_features <= 500:
    rows = permute(feature_cols, 20)
    method = f"permutation all {n_features} features n_repeats=20"
else:
    coef = get_coef()
    if coef is None:
        rows = []
        method = "unavailable: no reliable ranking for >500 features"
    else:
        top = [feature_cols[i] for i in np.argsort(coef)[::-1][:50]]
        rows = permute(top, 5)
        method = f"coef ranking permute top {len(top)} features n_repeats=5"

imp = pd.DataFrame(rows, columns=['feature', 'importance_mean', 'importance_sd'])
imp = imp.sort_values('importance_mean', ascending=False).head(50).reset_index(drop=True)
for c in ['importance_mean', 'importance_sd']:
    imp[c] = imp[c].apply(lambda x: float(f"{x:.4g}") if pd.notna(x) else x)

mlca_runtime.save_table(imp, "interpretation.csv", caption="Feature importance ranking")
print(f"method: {method}")
if len(imp):
    top = imp.iloc[0]['feature']
    mlca_runtime.record_fact_candidate("top_feature", str(top), source="interpretation.csv")
    print(f"top_feature={top} importance={imp.iloc[0]['importance_mean']}")
