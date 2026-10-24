import mlca_runtime, json, pickle
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

def final_est(m):
    if isinstance(m, Pipeline):
        return m.named_steps[list(m.named_steps)[-1]]
    return m

def get_cols(m):
    if hasattr(m, 'feature_names_in_'):
        return list(m.feature_names_in_)
    if isinstance(m, Pipeline):
        for step in m.named_steps.values():
            if hasattr(step, 'feature_names_in_'):
                return list(step.feature_names_in_)
    return None

def fallback_cols(val, protocol):
    targets = set(protocol.get('targets', []))
    ex = protocol.get('excluded_columns', [])
    if isinstance(ex, dict):
        ex = ex.get('columns', []) or []
    ex = set(ex) | targets | ({protocol.get('group_column')} if protocol.get('group_column') else set())
    cols = []
    for c in val.columns:
        if c in ex:
            continue
        s = val[c]
        if s.nunique(dropna=False) == len(val) and (s.dtype.kind in 'iuO'):
            continue
        cols.append(c)
    return cols

def coef_importance(m, raw_cols):
    est = final_est(m)
    if hasattr(est, 'coef_'):
        v = np.asarray(est.coef_)
        if v.ndim > 1:
            v = np.linalg.norm(v, axis=0) if v.shape[0] > 1 else v[0]
        v = np.abs(v).ravel()
    elif hasattr(est, 'feature_importances_'):
        v = np.abs(np.asarray(est.feature_importances_)).ravel()
    else:
        return None
    if len(v) == len(raw_cols):
        return pd.Series(v, index=raw_cols)
    if isinstance(m, Pipeline):
        for step in m.named_steps.values():
            if step is est or not hasattr(step, 'get_support') or not hasattr(step, 'feature_names_in_'):
                continue
            try:
                support = step.get_support()
            except Exception:
                continue
            names = np.asarray(step.feature_names_in_)
            if len(support) == len(names):
                sel = names[support]
                if len(sel) == len(v) and set(sel).issubset(set(raw_cols)):
                    return pd.Series(v, index=sel)
    return None

def score(m, X, y):
    if hasattr(m, 'predict_proba'):
        p = m.predict_proba(X)
    else:
        d = m.decision_function(X)
        p = np.column_stack([-d, d]) if np.ndim(d) == 1 else d
    if p.shape[1] == 2:
        classes = getattr(m, 'classes_', None)
        if classes is None:
            classes = getattr(final_est(m), 'classes_', None)
        yb = (np.asarray(y) == classes[1]).astype(int) if classes is not None else y
        return roc_auc_score(yb, p[:, 1])
    return roc_auc_score(y, p, multi_class='ovr', average='macro')

def perm_importance(m, X, y, cols, n_repeats, rng, base):
    rows = []
    for col in cols:
        orig = X[col].to_numpy(copy=True)
        Xp = X.copy()
        vals = []
        for _ in range(n_repeats):
            Xp[col] = rng.permutation(orig)
            vals.append(base - score(m, Xp, y))
        rows.append((col, float(np.mean(vals)), float(np.std(vals, ddof=0))))
    return pd.DataFrame(rows, columns=['feature', 'importance_mean', 'importance_sd']).sort_values('importance_mean', ascending=False)

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp['_protocol'].read_text())
train = pd.read_parquet(inp['train'])
val = pd.read_parquet(inp['val'])
target = protocol['targets'][0]
with open(inp['latest_model'], 'rb') as f:
    model = pickle.load(f)
if isinstance(model, dict):
    for k in ('model', 'pipeline', 'estimator'):
        if k in model:
            model = model[k]
            break
feature_cols = get_cols(model) or fallback_cols(val, protocol)
missing = [c for c in feature_cols if c not in val.columns]
if missing:
    raise KeyError(f"missing feature columns: {missing[:5]}")
X = val[feature_cols].copy()
y = val[target]
base = score(model, X, y)
mlca_runtime.record_metric("baseline_roc_auc", float(base), "val")
rng = np.random.default_rng(42)
nfeat = len(feature_cols)
if nfeat <= 500:
    perm_cols, n_repeats = feature_cols, 20
    method = "validation_permutation_all"
else:
    imp_series = coef_importance(model, feature_cols)
    if imp_series is None or len(imp_series) == 0:
        top = pd.DataFrame([{'feature': 'unavailable', 'importance_mean': np.nan, 'importance_sd': np.nan}])
        mlca_runtime.save_table(top, "interpretation.csv", caption="Feature importance unavailable")
        mlca_runtime.record_fact_candidate("top_feature", "unavailable", source="interpretation.csv")
        print("feature ranking unavailable")
        raise SystemExit(0)
    perm_cols = list(imp_series.sort_values(ascending=False).head(50).index)
    n_repeats = 5
    method = "validation_permutation_top50_by_model_importance"
print(f"train rows={len(train)} val rows={len(val)}")
print(f"method={method} features={len(perm_cols)} repeats={n_repeats}")
mlca_runtime.record_metric("n_permuted_features", float(len(perm_cols)), "val")
mlca_runtime.record_metric("permutation_repeats", float(n_repeats), "val")
imp = perm_importance(model, X, y, perm_cols, n_repeats, rng, base)
top = imp.head(15).reset_index(drop=True).round({'importance_mean': 4, 'importance_sd': 4})
mlca_runtime.save_table(top, "interpretation.csv", caption="Validation permutation importance")
top_feature = str(top.loc[0, 'feature']) if len(top) else "unavailable"
mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")
print(f"top_feature={top_feature}")
