import mlca_runtime, json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
if "targets" not in protocol:
    raise KeyError(f"missing targets; available: {list(protocol)}")
targets = protocol["targets"]
excluded = protocol.get("excluded_columns", [])
group = protocol.get("group_column")

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

def keylike(s):
    if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
        return False
    return s.notna().all() and s.nunique(dropna=True) == len(s)

drop = set(targets) | set(excluded) | ({group} if group else set())
feature_cols = [c for c in train.columns if c not in drop and not keylike(train[c])]


def importance_from(est):
    if est is None:
        return None, None, None
    if hasattr(est, "coef_"):
        co = np.asarray(est.coef_)
        if co.ndim == 1:
            co = co.reshape(1, -1)
        if co.ndim == 2 and co.shape[1] > 0:
            return np.abs(co).mean(axis=0), np.abs(co).std(axis=0), "coef_"
    if hasattr(est, "feature_importances_"):
        fi = np.asarray(est.feature_importances_).ravel()
        if fi.size > 0:
            return fi, np.zeros_like(fi), "feature_importances_"
    return None, None, None


def compute_names_from_model(m):
    names = list(getattr(m, "feature_names_in_", feature_cols))
    steps = m.steps[:-1] if isinstance(m, Pipeline) else []
    for _, step in steps:
        if hasattr(step, "get_support"):
            try:
                sup = list(step.get_support(indices=True))
            except Exception:
                continue
            in_names = list(getattr(step, "feature_names_in_", names))
            if not in_names:
                in_names = names
            if sup and max(sup) < len(in_names):
                names = [in_names[i] for i in sup]
        elif hasattr(step, "get_feature_names_out"):
            try:
                out_names = list(step.get_feature_names_out(names))
                if len(out_names) > 0:
                    names = out_names
            except Exception:
                pass
    return names


model = imp = sd = names_out = None
method = ""
candidates = [
    ("latest", inputs["latest_model"]),
    ("logistic", inputs["07_model_train_logistic_6BHE1H1F/model.pkl"]),
    ("alternative", inputs["08_model_train_alternative_SGR2R7DJ/alternative_model.pkl"]),
]
for tag, path in candidates:
    try:
        with open(path, "rb") as f:
            m = pickle.load(f)
    except Exception:
        continue
    final = m.steps[-1][1] if isinstance(m, Pipeline) else m
    i, s, meth = importance_from(final)
    if i is None:
        i, s, meth = importance_from(m)
    if i is not None:
        model, imp, sd = m, i, s
        method = f"frozen_{tag}_{meth}"
        names_out = compute_names_from_model(m)
        break

if imp is None:
    with open(inputs["06_features_select_features_9JJFXST3/selector.pkl"], "rb") as f:
        sel = pickle.load(f)
    i, s, meth = importance_from(sel)
    if i is None and hasattr(sel, "scores_"):
        i = np.abs(np.asarray(sel.scores_))
        s = np.zeros_like(i)
        meth = "scores_"
    if i is not None:
        imp, sd = i, s
        model = sel
        method = "selector_" + meth
        names_out = list(getattr(sel, "feature_names_in_", feature_cols))

if imp is None:
    raise RuntimeError("no reliable frozen feature importance available from declared models/selector")

raw_names = list(getattr(model, "feature_names_in_", feature_cols))
if not raw_names:
    raw_names = list(feature_cols)

if len(imp) != len(names_out):
    if len(imp) == len(feature_cols):
        names_out = list(feature_cols)
    elif len(imp) == len(raw_names):
        names_out = list(raw_names)
    else:
        names_out = [str(i) for i in range(len(imp))]


def to_raw(name):
    if name in feature_cols:
        return name
    s = str(name)
    for sep in ("__", "."):
        if sep in s and s.split(sep)[-1] in feature_cols:
            return s.split(sep)[-1]
    return None


agg = {}
for name, m_, s_ in zip(names_out, imp, sd):
    rn = to_raw(name)
    if rn is None:
        continue
    if rn in agg:
        agg[rn][0] += float(m_)
        agg[rn][1] = float(np.hypot(agg[rn][1], float(s_)))
        agg[rn][2] += 1
    else:
        agg[rn] = [float(m_), float(s_), 1]
rows = [(k, v[0] / v[2], v[1] / v[2]) for k, v in agg.items()]
if len(rows) < 15:
    rows = [(str(n), float(m_), float(s_)) for n, m_, s_ in zip(names_out, imp, sd)]

rows = sorted(rows, key=lambda r: r[1], reverse=True)[:15]
out = pd.DataFrame(rows, columns=["feature", "importance_mean", "importance_sd"])
for c in ["importance_mean", "importance_sd"]:
    out[c] = out[c].map(lambda x: float(f"{x:.4g}"))
mlca_runtime.save_table(out, "interpretation.csv", caption="Top 15 frozen model features")
top_feature = str(out.iloc[0]["feature"])
mlca_runtime.record_fact_candidate("top_feature", top_feature, source="interpretation.csv")
mlca_runtime.record_metric("n_input_features", float(len(feature_cols)), "val")
mlca_runtime.record_metric("n_ranked_features", float(len(imp)), "val")
mlca_runtime.record_metric("n_importance_repeats", 0.0, "val")
print(f"train={len(train)} val={len(val)} features={len(feature_cols)}")
print(f"method={method} ranked={len(imp)} repeats=0")
print(f"top_feature={top_feature} importance={out.iloc[0]['importance_mean']}")
