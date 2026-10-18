import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp['_protocol'].read_text())
val = pd.read_parquet(inp['val'])

targets = list(proto.get('targets') or [])
excluded = list(proto.get('excluded_columns') or [])
group_col = proto.get('group_column')
primary = proto['task']['primary_metric']
higher_better = primary in ('silhouette', 'calinski_harabasz', 'adjusted_rand_index')


def feat_cols(df):
    out = []
    for c in df.columns:
        if c in targets or c in excluded or c == group_col:
            continue
        s = df[c]
        if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and s.nunique() == df.shape[0]:
            continue
        out.append(c)
    return out


feat = feat_cols(val)
X_val = val[feat].copy()

pre = pickle.load(open(inp['05_preprocess_fit_preprocessing_A5A2YZJ3/preprocessor.pkl'], 'rb'))
sel = pickle.load(open(inp['06_features_retain_features_297WVMMN/selector_all.pkl'], 'rb'))
base = pickle.load(open(inp['07_model_train_baseline_FB68KDQ5/model.pkl'], 'rb'))
alt = pickle.load(open(inp['08_model_train_alternative_A16CW8P2/alternative_model.pkl'], 'rb'))


def get_cols(o):
    return list(o.feature_names_in_) if hasattr(o, 'feature_names_in_') else None


def sel_df(X, cols):
    return X[cols] if cols and all(c in X.columns for c in cols) else X


def prep(X):
    Xt = pre.transform(sel_df(X, get_cols(pre)))
    if sel is not None:
        scols = get_cols(sel)
        if not isinstance(Xt, pd.DataFrame) and scols and getattr(Xt, 'shape', [0, 0])[1] == len(scols):
            Xt = pd.DataFrame(Xt, columns=scols)
        Xt = sel.transform(Xt)
    return Xt


def feat_lab(model, X):
    if isinstance(model, Pipeline) and len(list(model.named_steps)) > 1:
        steps = list(model.named_steps.items())
        front = Pipeline(steps[:-1])
        X1 = sel_df(X, get_cols(front))
        return front.transform(X1), model.predict(X1)
    Xf = prep(X)
    return Xf, model.predict(Xf)


def safe(fn, *args, **kw):
    try:
        return float(fn(*args, **kw))
    except Exception:
        return float('nan')


rows, SC = [], {}
for name, mdl in [('baseline', base), ('alternative', alt)]:
    Xf, lab = feat_lab(mdl, X_val)
    Xf = np.asarray(Xf)
    lab = np.asarray(lab)
    n = len(Xf)
    if n > 10000:
        idx = np.random.RandomState(42).choice(n, 10000, replace=False)
        Xs, ls = Xf[idx], lab[idx]
    else:
        Xs, ls = Xf, lab
    s = safe(silhouette_score, Xs, ls, random_state=42)
    c = safe(calinski_harabasz_score, Xf, lab)
    d = safe(davies_bouldin_score, Xf, lab)
    SC[name] = s if not np.isnan(s) else -1.0
    rows.append({'model': name, 'metric': primary, 'value': round(s, 4)})
    rows.append({'model': name, 'metric': 'calinski_harabasz', 'value': round(c, 4)})
    rows.append({'model': name, 'metric': 'davies_bouldin', 'value': round(d, 4)})

vdf = pd.DataFrame(rows)
mlca_runtime.save_table(vdf, 'validation_metrics.csv', caption='Validation clustering metrics per candidate')

best_name = (max if higher_better else min)(SC, key=SC.get)
best_val = SC[best_name]
mlca_runtime.record_metric(primary, best_val, split='val')
mlca_runtime.record_fact_candidate('best_validation_score', best_val, source='validation_metrics.csv')
mlca_runtime.record_fact_candidate('best_model', best_name, source='validation_metrics.csv')

print(f"val {X_val.shape}, features {len(feat)}")
print(f"sil baseline={SC['baseline']:.4f} alt={SC['alternative']:.4f}")
print(f"best={best_name} {primary}={best_val:.4f}")
