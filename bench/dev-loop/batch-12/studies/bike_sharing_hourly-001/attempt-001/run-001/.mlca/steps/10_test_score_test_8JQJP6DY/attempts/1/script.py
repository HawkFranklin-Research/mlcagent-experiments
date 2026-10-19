import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())

for key in ["targets", "excluded_columns", "group_column", "task"]:
    if key not in proto:
        raise KeyError(f"Missing _protocol[{key!r}]; available keys: {list(proto.keys())}")

targets = list(proto["targets"])
excluded = set(proto["excluded_columns"]) | set(targets)
group_col = proto["group_column"]
if group_col is not None:
    excluded.add(group_col)

train = pd.read_parquet(inp["train"])

def is_keylike(s):
    if s.dtype == object or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return s.nunique(dropna=False) == len(s)
    if pd.api.types.is_integer_dtype(s):
        return s.nunique(dropna=False) == len(s)
    return False

feature_cols = [c for c in train.columns if c not in excluded and not is_keylike(train[c])]

pm = str(proto["task"]["primary_metric"])
vm = pd.read_csv(inp["09_validate_score_validation_51H7Y4ST/validation_metrics.csv"])
vrows = vm[vm["metric"].astype(str).str.lower() == pm.lower()].copy()
vrows["value"] = pd.to_numeric(vrows["value"], errors="coerce")
vrows = vrows.dropna(subset=["value"])
if vrows.empty:
    raise ValueError(f"No validation rows for primary metric {pm!r}")
lower_better = pm.lower() in {"rmse", "mae", "mse", "log_loss", "mape"}
best_row = vrows.sort_values("value", ascending=lower_better).iloc[0]
best_label = str(best_row["model"])

def load_pkl(key):
    with open(inp[key], "rb") as f:
        return pickle.load(f)

pre = load_pkl("05_preprocess_fit_preprocessing_7SHSFSVP/preprocessor.pkl")
sel = load_pkl("06_features_retain_features_JQTP2HCD/selector_all.pkl")
model_paths = {
    "baseline": "07_model_train_baseline_VKCYA88G/model.pkl",
    "alternative": "08_model_train_alternative_65RN6RC0/alternative_model.pkl",
}
models = {k: load_pkl(v) for k, v in model_paths.items()}

def match_model(label):
    l = label.lower()
    for name, path in model_paths.items():
        d = inp[path].parent.name.lower()
        if d in l or l in d or name in l:
            return name
    if "baseline" in l:
        return "baseline"
    if "alternative" in l or "latest" in l:
        return "alternative"
    raise ValueError(f"Cannot map validation model {label!r}")

selected_name = match_model(best_label)
selected_model = models[selected_name]

test = pd.read_parquet(inp["test"])
X_raw = test[feature_cols].copy()

def feature_names(obj):
    return list(getattr(obj, "feature_names_in_", []))

def direct_predict(model, raw):
    ns = feature_names(model)
    if ns and all(c in raw.columns for c in ns):
        return model.predict(raw[ns])
    return None

def to_frame(x, model, index):
    ns = feature_names(model)
    if isinstance(x, pd.DataFrame):
        if ns and set(ns).issubset(x.columns):
            return x[ns]
        if ns and x.shape[1] == len(ns):
            x = x.copy()
            x.columns = ns
        return x
    if ns and getattr(x, "shape", [None, None])[1] == len(ns):
        if hasattr(x, "toarray"):
            x = x.toarray()
        return pd.DataFrame(x, columns=ns, index=index)
    return x

def frontend_predict(model, raw):
    if sel is not None:
        ns = feature_names(sel)
        if ns and all(c in raw.columns for c in ns):
            z = sel.transform(raw[ns])
        else:
            pns = feature_names(pre)
            if not pns or not all(c in raw.columns for c in pns):
                raise ValueError("No saved frontend accepts the raw feature columns")
            z = pre.transform(raw[pns])
            z = sel.transform(z)
    else:
        pns = feature_names(pre)
        if not pns or not all(c in raw.columns for c in pns):
            raise ValueError("No saved frontend accepts the raw feature columns")
        z = pre.transform(raw[pns])
    return model.predict(to_frame(z, model, raw.index))

pred = direct_predict(selected_model, X_raw)
if pred is None:
    pred = frontend_predict(selected_model, X_raw)
pred = np.asarray(pred)
if pred.ndim == 1:
    pred = pred.reshape(-1, 1)

rows = []
per_target = {"rmse": [], "mae": [], "r2": []}
for j, target in enumerate(targets):
    y = test[target].to_numpy()
    p = pred[:, j] if pred.shape[1] > j else pred[:, 0]
    r = float(np.sqrt(mean_squared_error(y, p)))
    m = float(mean_absolute_error(y, p))
    r2 = float(r2_score(y, p))
    per_target["rmse"].append(r)
    per_target["mae"].append(m)
    per_target["r2"].append(r2)
    for name, value in [("rmse", r), ("mae", m), ("r2", r2)]:
        rows.append({"model": best_label, "metric": f"{name}[{target}]", "value": value})

macro = {name: float(np.mean(vals)) for name, vals in per_target.items()}
for name, value in macro.items():
    rows.append({"model": best_label, "metric": f"{name}_macro", "value": value})

metrics_df = pd.DataFrame(rows)
metrics_df["value"] = metrics_df["value"].map(lambda v: float(f"{v:.4g}"))
mlca_runtime.save_table(metrics_df, "test_metrics.csv", caption="Test metrics for validation-selected candidate")

primary_value = float(macro[pm])
mlca_runtime.record_metric(pm, primary_value, "test")
mlca_runtime.record_fact_candidate("test_score", primary_value, source="test_metrics.csv")
for name, value in macro.items():
    mlca_runtime.record_metric(name, value, "test")

print(f"train={train.shape} test={test.shape}")
print(f"selected={selected_name} primary={pm} test={primary_value:.4g}")
