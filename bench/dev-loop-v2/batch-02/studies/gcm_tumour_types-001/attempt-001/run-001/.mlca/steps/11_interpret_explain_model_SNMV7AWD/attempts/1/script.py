import mlca_runtime, json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
if "targets" not in protocol:
    raise KeyError(f"missing targets; available keys: {list(protocol)}")
targets = protocol["targets"]
excluded = protocol.get("excluded_columns", [])
group = protocol.get("group_column")
task = protocol["task"]
metric = task["primary_metric"]
y_col = targets[0]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

def keylike(s):
    if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
        return False
    return s.notna().all() and s.nunique(dropna=True) == len(s)

drop = set(targets) | set(excluded) | ({group} if group else set())
feature_cols = [c for c in train.columns if c not in drop and not keylike(train[c])]

with open(inputs["latest_model"], "rb") as f:
    model = pickle.load(f)
raw_names = [c for c in list(getattr(model, "feature_names_in_", feature_cols)) if c in train.columns]
if not raw_names:
    raw_names = feature_cols
missing = [c for c in raw_names if c not in val.columns]
if missing:
    raise KeyError(f"model features missing from val: {missing[:5]}")
X_val = val[raw_names]

def importance_from(est):
    if hasattr(est, "coef_"):
        co = np.asarray(est.coef_)
        if co.ndim == 1:
            co = co.reshape(1, -1)
        return np.abs(co).mean(axis=0), np.abs(co).std(axis=0)
    if hasattr(est, "feature_importances_"):
        fi = np.asarray(est.feature_importances_)
        return fi, np.zeros_like(fi)
    return None, None

final = model.steps[-1][1] if isinstance(model, Pipeline) else model
names_out = list(raw_names)
for _, step in (model.steps[:-1] if isinstance(model, Pipeline) else []):
    if hasattr(step, "get_support"):
        sup = step.get_support(indices=True)
        in_names = list(getattr(step, "feature_names_in_", names_out))
        names_out = [in_names[i] for i in sup]
    elif hasattr(step, "get_feature_names_out"):
        try:
            names_out = list(step.get_feature_names_out(names_out))
        except Exception:
            pass

imp, sd = importance_from(final)
method = "frozen estimator coefficients" if imp is not None else ""
if imp is None:
    with open(inputs["06_features_select_features_9JJFXST3/selector.pkl"], "rb") as f:
        sel = pickle.load(f)
    imp, sd = importance_from(sel)
    names_out = list(getattr(sel, "feature_names_in_", raw_names))
    if imp is None and hasattr(sel, "scores_"):
        imp = np.abs(np.asarray(sel.scores_))
        sd = np.zeros_like(imp)
    method = "saved selector scores"
if imp is None:
    raise RuntimeError("no reliable frozen feature importance available")

if len(imp) != len(names_out):
    if len(imp) == len(raw_names):
        names_out = list(raw_names)
    elif len(imp) == len(feature_cols):
        names_out = list(feature_cols)
    else:
        names_out = [str(i) for i in range(len(imp))]

def to_raw(name):
    if name in raw_names:
        return name
    s = str(name)
    for sep in ("__", "."):
        if sep in s and s.split(sep)[-1] in raw_names:
            return s.split(sep)[-1]
    return None

agg = {}
for name, m, s in zip(names_out, imp, sd):
    rn = to_raw(name)
    if rn is None:
        continue
    if rn in agg:
        agg[rn][0] += float(m)
        agg[rn][1] = float(np.hypot(agg[rn][1], float(s)))
        agg[rn][2] += 1
    else:
        agg[rn] = [float(m), float(s), 1]
rows = [(k, v[0] / v[2], v[1] / v[2]) for k, v in agg.items()]
if len(rows) < 15:
    rows = [(str(n), float(m), float(s)) for n, m, s in zip(names_out, imp, sd)]

rows = sorted(rows, key=lambda r: r[1], reverse=True)[:15]
out = pd.DataFrame(rows, columns=["feature", "importance_mean", "importance_sd"])
for c in ["importance_mean", "importance_sd"]:
    out[c] = out[c].map(lambda x: float(f"{x:.4g}"))
mlca_runtime.save_table(out, "interpretation.csv", caption="Top 15 frozen model features")
top_feature = str(out.iloc[0]["feature"])
mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")
mlca_runtime.record_metric("n_input_features", float(len(raw_names)), "val")
mlca_runtime.record_metric("n_ranked_features", float(len(imp)), "val")
mlca_runtime.record_metric("n_importance_repeats", 0.0, "val")
print(f"train={len(train)} val={len(val)} features={len(raw_names)}")
print(f"importance={method}; ranked={len(imp)}; repeats=0")
print(f"top_feature={top_feature} importance={out.iloc[0]['importance_mean']}")
