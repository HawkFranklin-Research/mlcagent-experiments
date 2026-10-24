import json, pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.inspection import permutation_importance

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol.get("group_column")
task = protocol["task"]

def excluded_set(ex):
    if isinstance(ex, (list, tuple, set)):
        return set(ex)
    if isinstance(ex, dict):
        out = set()
        for v in ex.values():
            if isinstance(v, (list, tuple, set)):
                out.update(v)
        return out
    return set()

ex_set = excluded_set(excluded)
n_classes = task.get("n_classes", 2)
metric = task.get("primary_metric")
if task.get("type") == "multiclass" or n_classes > 2:
    scoring = "roc_auc_ovr"
else:
    scoring = "roc_auc"
if metric in {"roc_auc", "roc_auc_ovr", "roc_auc_ovr_weighted"}:
    scoring = metric
elif metric == "auc":
    scoring = "roc_auc"

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
with open(inp["latest_model"], "rb") as f:
    model = pickle.load(f)

if hasattr(model, "feature_names_in_"):
    feature_cols = list(model.feature_names_in_)
else:
    def keylike(s):
        if s.dtype == object:
            return s.nunique(dropna=True) == len(s) and len(s) > 0
        if pd.api.types.is_integer_dtype(s.dtype):
            return s.nunique(dropna=True) == len(s) and len(s) > 0
        return False
    feature_cols = [c for c in val.columns
                    if c not in set(targets) and c not in ex_set
                    and c != group and not keylike(val[c])]

missing = [c for c in feature_cols if c not in val.columns]
if missing:
    raise ValueError(f"Missing frozen-model input columns in val: {missing[:5]}")
bad = set(feature_cols) & (set(targets) | ex_set)
if bad:
    raise ValueError(f"Target/excluded columns in frozen-model inputs: {sorted(bad)[:5]}")

target = targets[0]
X_val = val.loc[:, feature_cols].copy()
y_val = val[target].copy()
if hasattr(model, "classes_"):
    cls = list(model.classes_)
    u = set(pd.unique(y_val.dropna()))
    if u.issubset(set(cls)):
        y_val = y_val.map({c: i for i, c in enumerate(cls)})
mask = y_val.notna()
X_val = X_val.loc[mask]
y_val = y_val.loc[mask]

n_features = len(feature_cols)
n_repeats = 0
method = ""
out = pd.DataFrame(columns=["feature", "importance_mean", "importance_sd"])

if n_features <= 500:
    n_repeats = 20
    method = "validation_permutation_importance"
    imp = permutation_importance(model, X_val, y_val, scoring=scoring,
                                 n_repeats=n_repeats, random_state=42, n_jobs=1)
    idx = np.argsort(imp.importances_mean)[::-1][:15]
    out = pd.DataFrame({
        "feature": np.array(feature_cols)[idx],
        "importance_mean": imp.importances_mean[idx],
        "importance_sd": imp.importances_std[idx],
    })
else:
    est = list(model.named_steps.values())[-1] if hasattr(model, "named_steps") and model.named_steps else model
    vals = None
    kind = ""
    if hasattr(est, "coef_"):
        vals = np.abs(np.ravel(est.coef_))
        kind = "coef_abs"
    elif hasattr(est, "feature_importances_"):
        vals = np.abs(np.ravel(est.feature_importances_))
        kind = "feature_importances_abs"
    if vals is not None and len(vals) != len(feature_cols) and hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if hasattr(step, "get_support"):
                sup = step.get_support()
                if len(sup) == len(feature_cols) and len(vals) == int(sup.sum()):
                    vals = pd.Series(vals, index=np.array(feature_cols)[sup]).reindex(feature_cols).fillna(0).values
                    break
    if vals is not None and len(vals) == len(feature_cols):
        order = np.argsort(vals)[::-1][:15]
        out = pd.DataFrame({
            "feature": np.array(feature_cols)[order],
            "importance_mean": vals[order],
            "importance_sd": np.nan,
        })
        method = f"model_{kind}"
    else:
        method = "unavailable_reliable_ranking_no_full_width_permutation"

for c in ["importance_mean", "importance_sd"]:
    out[c] = out[c].apply(lambda x: float(f"{x:.4g}") if pd.notna(x) else np.nan)

caption = (f"method={method}; scoring={scoring}; n_features={n_features}; "
           f"n_repeats={n_repeats}; split=val; rows={len(X_val)}")
mlca_runtime.save_table(out, "interpretation.csv", caption=caption)

if len(out):
    top = out.iloc[0]
    mlca_runtime.record_fact_candidate("top_feature", str(top["feature"]), source="interpretation.csv")
    print(f"train rows={len(train)}, val rows={len(X_val)}, features={n_features}, method={method}, repeats={n_repeats}")
    print(f"top feature: {top['feature']} ({top['importance_mean']:.4g})")
else:
    print(f"train rows={len(train)}, val rows={len(X_val)}, features={n_features}, no reliable ranking")
