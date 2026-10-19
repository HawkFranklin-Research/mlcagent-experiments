import mlca_runtime
import json, pickle, numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
prot = json.loads(inp["_protocol"].read_text())
targets = list(prot["targets"])
excluded = list(prot["excluded_columns"])
group_col = prot.get("group_column")
primary = prot["task"]["primary_metric"]
train = pd.read_parquet(inp["train"])
test = pd.read_parquet(inp["test"])

base = [c for c in train.columns if c not in set(targets + excluded + ([group_col] if group_col else []))]
feat = []
for c in base:
    s = train[c]
    low = str(c).lower()
    if (s.dtype == object or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype)):
        if s.nunique(dropna=False) == len(s):
            continue
    if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=False) == len(s):
        if low in ("id", "index", "key") or low.endswith("_id") or low.startswith("id_"):
            continue
    feat.append(c)

def read_vm():
    p = inp["09_validate_score_validation_GKTQ5B1P/validation_metrics.parquet"]
    try:
        return pd.read_parquet(p)
    except Exception:
        return pd.read_csv(inp["09_validate_score_validation_GKTQ5B1P/validation_metrics.csv"])

vm = read_vm()
cols = {str(c).lower(): c for c in vm.columns}
lower_better = {"brier_score", "log_loss", "mse", "mae", "rmse"}
sel_name = None
if primary.lower() in cols:
    dc = vm
    for sc in ("split", "partition"):
        if sc in cols:
            dc = dc[dc[cols[sc]].astype(str).str.lower().isin(["val", "validation", "valid"])]
    if len(dc) == 0: dc = vm
    mcol = next((cols[k] for k in ("model", "model_name", "candidate", "estimator") if k in cols), None)
    if mcol:
        idx = dc[cols[primary.lower()]].astype(float).idxmin() if primary.lower() in lower_better else dc[cols[primary.lower()]].astype(float).idxmax()
        sel_name = str(dc.loc[idx, mcol])
else:
    mcol = next((cols[k] for k in ("metric", "metric_name", "name") if k in cols), None)
    vcol = next((cols[k] for k in ("value", "score", "val") if k in cols), None)
    if not (mcol and vcol): raise ValueError(f"cannot parse validation metrics columns: {list(vm.columns)}")
    dm = vm[vm[mcol].astype(str).str.lower() == primary.lower()]
    for sc in ("split", "partition"):
        if sc in cols:
            dm = dm[dm[cols[sc]].astype(str).str.lower().isin(["val", "validation", "valid"])]
    if len(dm) == 0: raise ValueError(f"primary metric {primary} not found in validation metrics")
    model_col = next((cols[k] for k in ("model", "model_name", "candidate", "estimator") if k in cols), None)
    idx = dm[vcol].astype(float).idxmin() if primary.lower() in lower_better else dm[vcol].astype(float).idxmax()
    if model_col: sel_name = str(dm.loc[idx, model_col])

def load(p):
    with open(p, "rb") as f: return pickle.load(f)

if sel_name and "logistic" in sel_name.lower():
    model_path = inp["07_model_train_logistic_G7CT94CC/model.pkl"]
elif sel_name and ("alternative" in sel_name.lower() or "hist" in sel_name.lower() or "boost" in sel_name.lower()):
    model_path = inp["08_model_train_alternative_PSET1F82/alternative_model.pkl"]
else:
    model_path = inp["latest_model"]

pre = load(inp["05_preprocess_fit_preprocessing_MQJ4AQ8S/preprocessor.pkl"])
sel = load(inp["06_features_select_features_ZWFAMHT4/selector.pkl"])
model = load(model_path)

def names(obj):
    try: return list(obj.feature_names_in_)
    except Exception: return None

def accepts_raw(obj):
    n = names(obj)
    return n is None or n == feat

def pipe(steps):
    steps = [(n, o) for n, o in steps if o is not None]
    return Pipeline(steps) if len(steps) > 1 else steps[0][1]

cands = [model]
cands.append(pipe([("pre", pre), ("sel", sel), ("model", model)]))
cands.append(pipe([("sel", sel), ("model", model)]))
cands.append(pipe([("pre", pre), ("model", model)]))
Xchk = train[feat].iloc[[0]]
pipeline = None
for cand in cands:
    if not hasattr(cand, "predict_proba") or not accepts_raw(cand): continue
    try:
        if cand.predict_proba(Xchk).shape[0] == 1:
            pipeline = cand
            break
    except Exception:
        pass
if pipeline is None: raise RuntimeError("no saved pipeline accepts the raw feature DataFrame")

def classes_of(obj):
    if hasattr(obj, "classes_"): return np.asarray(obj.classes_)
    if isinstance(obj, Pipeline): return np.asarray(obj.steps[-1][1].classes_)
    raise RuntimeError("model classes unavailable")

X = test[feat]
y = test[targets[0]].to_numpy()
proba = pipeline.predict_proba(X)
classes = classes_of(pipeline)
if list(classes) != list(pipeline.classes_) if hasattr(pipeline, "classes_") else False:
    raise RuntimeError("probability columns are not aligned with model classes")

def macro_ovr_auc(y_true, p, cls):
    vals = []
    for j, c in enumerate(cls):
        b = (y_true == c).astype(int)
        if np.unique(b).size == 2:
            vals.append(roc_auc_score(b, p[:, j]))
    return float(np.mean(vals)) if vals else np.nan

def metrics(y_true, p, cls):
    pred = cls[np.argmax(p, axis=1)]
    return macro_ovr_auc(y_true, p, cls), float(f1_score(y_true, pred, labels=cls, average="macro"))

auc_val, f1_val = metrics(y, proba, classes)
rng = np.random.default_rng(42)
auc_bs, f1_bs = [], []
n = len(y)
for _ in range(1000):
    idx = rng.integers(0, n, n)
    a, f = metrics(y[idx], proba[idx, :], classes)
    if not np.isnan(a): auc_bs.append(a)
    f1_bs.append(f)

def ci(vals):
    if not vals: return (np.nan, np.nan)
    return tuple(float(x) for x in np.nanpercentile(vals, [2.5, 97.5]))

auc_lo, auc_hi = ci(auc_bs)
f1_lo, f1_hi = ci(f1_bs)
is_auc = primary.lower() in ("roc_auc_ovr", "roc_auc")
if is_auc:
    pval, plo, phi, pname = auc_val, auc_lo, auc_hi, primary
    other = {"metric": "macro_f1", "value": f1_val, "ci_low": f1_lo, "ci_high": f1_hi}
else:
    pval, plo, phi, pname = f1_val, f1_lo, f1_hi, primary
    other = {"metric": "roc_auc_ovr", "value": auc_val, "ci_low": auc_lo, "ci_high": auc_hi}

def s4(x): return float(f"{x:.4g}") if pd.notna(x) else np.nan
rows = [{"metric": pname, "value": s4(pval), "ci_low": s4(plo), "ci_high": s4(phi)}, other]
for r in rows:
    for k in ("value", "ci_low", "ci_high"): r[k] = s4(r[k])
mlca_runtime.save_table(pd.DataFrame(rows), "test_metrics.csv", caption="Test metrics with 95% bootstrap CI")
mlca_runtime.record_metric(pname, pval, "test", ci=(plo, phi))
if is_auc:
    mlca_runtime.record_fact_candidate("test_roc_auc", auc_val, source="test_metrics.csv")
print(f"selected={sel_name or 'latest'}; n_test={len(y)}; {pname}={pval:.4f}; macro_f1={f1_val:.4f}")
