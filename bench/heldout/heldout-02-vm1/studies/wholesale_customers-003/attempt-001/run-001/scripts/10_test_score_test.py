import mlca_runtime, json, pickle
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
prot = json.loads(inp["_protocol"].read_text())
try:
    targets = prot["targets"]
    excluded = prot["excluded_columns"]
    group_col = prot["group_column"]
    task = prot["task"]
    primary = task["primary_metric"]
except KeyError as e:
    raise KeyError(f"Missing protocol key {e}; available top-level keys: {sorted(prot)}")
if not primary:
    raise ValueError("_protocol['task']['primary_metric'] is null")

train = pd.read_parquet(inp["train"])
test = pd.read_parquet(inp["test"])

def is_key_like(s):
    if not s.is_unique:
        return False
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return True
    if pd.api.types.is_integer_dtype(s):
        name = str(s.name).lower()
        vals = s.to_numpy()
        return any(t in name for t in ("id", "code", "key", "index", "row")) or np.array_equal(
            np.sort(vals), np.arange(vals.min(), vals.min() + len(vals))
        )
    return False

drop = set(targets) | set(excluded)
if group_col is not None:
    drop.add(group_col)
feat = [c for c in train.columns if c not in drop and not is_key_like(train[c])]

cand = {}
for key, label in [
    ("07_model_train_baseline_V5KHW25A/model.pkl", "baseline"),
    ("08_model_train_alternative_78714TF3/alternative_model.pkl", "alternative"),
    ("latest_model", "latest"),
]:
    with open(inp[key], "rb") as f:
        cand[label] = pickle.load(f)
with open(inp["05_preprocess_fit_preprocessing_HBN850GP/preprocessor.pkl"], "rb") as f:
    prep = pickle.load(f)
with open(inp["06_features_retain_features_5BEJHTH5/selector_all.pkl"], "rb") as f:
    sel = pickle.load(f)

vm = pd.read_csv(inp["09_validate_score_validation_Q3R7JTH4/validation_metrics.csv"])
low = {str(c).lower(): c for c in vm.columns}
mc = low.get("model") or low.get("candidate") or low.get("name")
metc = low.get("metric") or low.get("metric_name")
valc = low.get("value") or low.get("score")
if mc and metc and valc:
    long = vm[[mc, metc, valc]].rename(columns={mc: "model", metc: "metric", valc: "value"})
elif mc:
    long = vm.melt(id_vars=[mc], var_name="metric", value_name="value").rename(columns={mc: "model"})
else:
    idx = vm.index.name or "index"
    long = vm.reset_index().melt(id_vars=[idx], var_name="metric", value_name="value").rename(columns={idx: "model"})
p = long[long["metric"].astype(str).str.lower() == str(primary).lower()].copy()
if p.empty:
    raise KeyError(f"Validation metrics has no row for primary metric {primary}")
ascending = str(primary).lower() in ("davies_bouldin", "davies-bouldin")
p = p.groupby("model", as_index=False)["value"].mean().sort_values("value", ascending=ascending)
selected_name = str(p.iloc[0]["model"])

def find_model(name):
    if name in cand:
        return cand[name]
    nl = name.lower()
    for k, v in cand.items():
        if k.lower() in nl or nl in k.lower():
            return v
    raise KeyError(f"No saved model matches {name}; available: {list(cand)}")

model = find_model(selected_name)

def raw_input(m, df):
    cols = getattr(m, "feature_names_in_", None)
    cols = list(cols) if cols is not None else feat
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing model features: {missing}")
    return df[cols]

def transformed(m, X):
    if isinstance(m, Pipeline):
        try:
            return m[:-1].transform(X)
        except Exception:
            pass
    if hasattr(m, "transform") and not isinstance(m, Pipeline):
        try:
            return m.transform(X)
        except Exception:
            pass
    Z = X
    if prep is not None:
        cols = getattr(prep, "feature_names_in_", None)
        Z = prep.transform(X[list(cols)]) if cols is not None else prep.transform(X)
    if sel is not None:
        names = getattr(sel, "named_steps", {})
        if isinstance(sel, Pipeline) and any("preprocess" in str(k).lower() for k in names):
            cols = getattr(sel, "feature_names_in_", None)
            return sel.transform(X[list(cols)]) if cols is not None else sel.transform(X)
        Z = sel.transform(Z)
    return Z

Xtr = raw_input(model, train)
Xte = raw_input(model, test)
ytr = model.predict(Xtr)
yte = model.predict(Xte)
Xt = transformed(model, Xte)
n = len(Xte)
if n > 10000:
    idx = np.random.RandomState(42).choice(n, 10000, replace=False)
    Xs = Xt.iloc[idx] if hasattr(Xt, "iloc") else Xt[idx]
    ys = yte[idx]
else:
    Xs = Xt
    ys = yte

metrics = {
    "silhouette": float(silhouette_score(Xs, ys, sample_size=min(10000, len(ys)), random_state=42)),
    "calinski_harabasz": float(calinski_harabasz_score(Xs, ys)),
    "davies_bouldin": float(davies_bouldin_score(Xs, ys)),
}

def rs(v):
    return float(f"{v:.4g}")

mapl = {
    "silhouette": "silhouette",
    "calinski_harabasz": "calinski_harabasz",
    "calinski-harabasz": "calinski_harabasz",
    "davies_bouldin": "davies_bouldin",
    "davies-bouldin": "davies_bouldin",
}
pkey = mapl.get(str(primary).lower(), str(primary).lower())
if pkey not in metrics:
    raise KeyError(f"Unsupported primary metric {primary}")
pval = metrics[pkey]

rows = []
for k, v in metrics.items():
    rows.append({"model": selected_name, "metric": str(primary) if k == pkey else k, "value": rs(v)})
out = pd.DataFrame(rows)
mlca_runtime.save_table(out, "test_metrics.csv", caption="Test metrics for validation-selected clustering candidate")

for k, v in metrics.items():
    if k != pkey:
        mlca_runtime.record_metric(k, v, "test")
mlca_runtime.record_metric(str(primary), pval, "test")
mlca_runtime.record_fact_candidate("test_score", pval, source="test_metrics.csv")

assign = pd.concat([
    pd.DataFrame({"row_id": train.index, "partition": "train", "cluster": ytr}),
    pd.DataFrame({"row_id": test.index, "partition": "test", "cluster": yte}),
], ignore_index=True)
mlca_runtime.save_row_outputs(assign, "assignments.parquet")

print(f"selected={selected_name}")
print(f"test_rows={n} metric_rows={len(ys)}")
print(f"{primary}={pval:.4f}")
