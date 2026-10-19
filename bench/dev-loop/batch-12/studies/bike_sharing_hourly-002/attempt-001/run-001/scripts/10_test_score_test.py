import mlca_runtime, json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
targets = protocol["targets"]
excluded = list(protocol.get("excluded_columns", []))
group = protocol.get("group_column")
task = protocol["task"]
primary = task["primary_metric"]

train = pd.read_parquet(inp["train"])
test = pd.read_parquet(inp["test"])
val = pd.read_csv(inp["09_validate_score_validation_PTFC7YN8/validation_metrics.csv"])

def pick_col(df, names):
    for c in df.columns:
        if str(c).lower() in names:
            return c
    raise KeyError(f"missing {names}; columns={list(df.columns)}")

mcol = pick_col(val, {"model", "model_name", "candidate"})
metcol = pick_col(val, {"metric", "metric_name"})
vcol = pick_col(val, {"value", "metric_value", "score"})
tcol = next((c for c in val.columns if str(c).lower() in {"target", "target_name", "outcome"}), None)

mnames = val[metcol].astype(str).str.lower()
v = val[mnames == primary.lower()].copy()
if v.empty:
    v = val[mnames.str.startswith(primary.lower() + "_")].copy()
if v.empty:
    raise ValueError(f"metric {primary} not in val metrics: {sorted(set(mnames))[:10]}")
if tcol is not None:
    mask = v[tcol].astype(str).str.lower().isin({"macro", "mean", "average", "overall", "all"})
    if mask.any():
        v = v[mask]
grp = v.groupby(mcol, as_index=False)[vcol].mean()
if grp.empty:
    raise ValueError("empty validation aggregate")

lower_better = primary.lower() in {"rmse", "mae", "mse", "logloss", "hinge",
                                    "neg_mean_squared_error", "mean_squared_error"}
best = grp.sort_values(vcol, ascending=lower_better).iloc[0]
best_name = str(best[mcol])
best_val = float(best[vcol])

def model_path(name):
    s = name.lower()
    if "baseline" in s or "07_" in s or s in {"model", "baseline_model"}:
        return inp["07_model_train_baseline_GKA0GSZP/model.pkl"]
    if "alternative" in s or "08_" in s or "latest" in s:
        return inp["08_model_train_alternative_A4YVEKN6/alternative_model.pkl"]
    raise ValueError(f"unknown model {name}")

with open(model_path(best_name), "rb") as f:
    model = pickle.load(f)
with open(inp["05_preprocess_fit_preprocessing_T5EPBF1H/preprocessor.pkl"], "rb") as f:
    preproc = pickle.load(f)
with open(inp["06_features_retain_features_EWX96D8N/selector_all.pkl"], "rb") as f:
    selector = pickle.load(f)

def feature_columns(frame):
    cols, n = [], len(frame)
    for c in frame.columns:
        if c in targets or c in excluded or c == group:
            continue
        s = frame[c]
        is_id = s.nunique(dropna=False) == n and (
            pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)
            or pd.api.types.is_integer_dtype(s))
        if not is_id:
            cols.append(c)
    return cols

feature_cols = feature_columns(train)

def input_cols(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        for _, step in obj.steps:
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return None

def predict(obj, df):
    cols = input_cols(obj)
    if cols is not None and all(c in df.columns for c in cols):
        return np.asarray(obj.predict(df[cols])).ravel()
    if isinstance(obj, Pipeline):
        return np.asarray(obj.predict(df[feature_cols])).ravel()
    raw = input_cols(preproc) or feature_cols
    X = df[raw]
    sel_cols = input_cols(selector)
    if sel_cols is not None and set(sel_cols) == set(raw):
        Xt = selector.transform(X)
    else:
        Xt = selector.transform(preproc.transform(X))
    return np.asarray(obj.predict(Xt)).ravel()

y_true = test[targets].to_numpy()
y_pred = predict(model, test).reshape(-1, len(targets))
def rmse(a, b): return mean_squared_error(a, b) ** 0.5
fns = {"rmse": rmse, "mae": mean_absolute_error, "r2": r2_score}
if primary.lower() not in fns:
    raise ValueError(f"unsupported primary metric {primary}")
prim_fn = fns[primary.lower()]

rows = []
for j, t in enumerate(targets):
    rows += [(best_name, f"rmse_{t}", rmse(y_true[:, j], y_pred[:, j])),
             (best_name, f"mae_{t}", mean_absolute_error(y_true[:, j], y_pred[:, j])),
             (best_name, f"r2_{t}", r2_score(y_true[:, j], y_pred[:, j]))]
for name, fn in [("rmse", rmse), ("mae", mean_absolute_error), ("r2", r2_score)]:
    rows.append((best_name, f"{name}_macro",
                 float(np.mean([fn(y_true[:, j], y_pred[:, j]) for j in range(len(targets))]))))
rows.append((best_name, primary,
             float(np.mean([prim_fn(y_true[:, j], y_pred[:, j]) for j in range(len(targets))]))))

table = pd.DataFrame(rows, columns=["model", "metric", "value"])
table["value"] = table["value"].apply(lambda x: float(f"{float(x):.4g}"))
mlca_runtime.save_table(table, "test_metrics.csv", caption="Test metrics")
prim_val = float(table.loc[table["metric"] == primary, "value"].iloc[0])
mlca_runtime.record_metric(primary, prim_val, "test")
mlca_runtime.record_fact_candidate("test_score", prim_val, source="test_metrics.csv")
print(f"selected={best_name} val_{primary}={best_val:.4g}")
print(f"test rows={len(test)}")
print(f"test {primary}={prim_val:.4g}")
