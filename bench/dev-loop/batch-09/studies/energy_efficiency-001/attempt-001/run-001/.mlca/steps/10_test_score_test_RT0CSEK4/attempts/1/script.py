import mlca_runtime, json, pickle, numpy as np, pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())

def req(d, k):
    if k not in d:
        raise KeyError(f"Missing {k}; available top-level keys: {list(d.keys())}")
    return d[k]

targets = req(protocol, "targets")
excluded = req(protocol, "excluded_columns")
task = req(protocol, "task")
primary = req(task, "primary_metric")
group = protocol.get("group_column", None)

train = pd.read_parquet(paths["train"])
drop = set(targets) | set(excluded)
if group:
    drop.add(group)

feature_cols = []
for c in train.columns:
    if c in drop:
        continue
    s = train[c]
    is_text = pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype)
    if is_text and s.nunique(dropna=False) == len(s):
        continue
    if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=False) == len(s):
        continue
    feature_cols.append(c)

for t in targets:
    if t not in train.columns:
        raise KeyError(f"Target {t} missing from train")

val = pd.read_csv(paths["09_validate_score_validation_N590X4VJ/validation_metrics.csv"])
def find_col(df, cands):
    for c in cands:
        if c in df.columns:
            return c
    raise KeyError(f"None of {cands} in {list(df.columns)}")

model_col = find_col(val, ["model", "Model", "candidate", "Candidate", "model_name"])
metric_col = find_col(val, ["metric", "Metric", "metric_name"])
value_col = find_col(val, ["value", "Value", "score"])
target_col = "target" if "target" in val.columns else ("Target" if "Target" in val.columns else None)

mask = val[metric_col].astype(str).str.strip().str.lower() == str(primary).lower()
if not mask.any():
    mask = val[metric_col].astype(str).str.strip().str.lower().str.contains(str(primary).lower(), regex=False)
if not mask.any():
    raise ValueError(f"Primary metric {primary} not found in validation metrics")

sub = val.loc[mask].copy()
sub = sub.groupby(model_col, as_index=False)[value_col].mean()
lower_better = str(primary).lower() in {"rmse", "mae", "mse", "logloss", "mape", "rmsle"}
sub = sub.sort_values(value_col, ascending=lower_better)
selected_name = str(sub.iloc[0][model_col])
name_l = selected_name.lower()

if "baseline" in name_l:
    model_path = paths["07_model_train_baseline_Q7Z8BHZ9/model.pkl"]
elif "alternative" in name_l or "latest" in name_l:
    model_path = paths["08_model_train_alternative_5G4S54TV/alternative_model.pkl"]
elif "07" in name_l:
    model_path = paths["07_model_train_baseline_Q7Z8BHZ9/model.pkl"]
elif "08" in name_l:
    model_path = paths["08_model_train_alternative_5G4S54TV/alternative_model.pkl"]
else:
    raise ValueError(f"Cannot map selected validation candidate: {selected_name}")

test = pd.read_parquet(paths["test"])
for t in targets:
    if t not in test.columns:
        raise KeyError(f"Target {t} missing from test")

X_test = test[feature_cols].copy()
y_test = test[targets].copy()

with open(model_path, "rb") as f:
    model = pickle.load(f)
if not hasattr(model, "predict") and isinstance(model, dict) and "model" in model:
    model = model["model"]
if hasattr(model, "feature_names_in_"):
    expected = list(model.feature_names_in_)
    missing = [c for c in expected if c not in X_test.columns]
    if missing:
        raise KeyError(f"Missing model features: {missing}")
    X_test = X_test[expected]

pred = np.asarray(model.predict(X_test))
if pred.ndim == 1:
    pred = pred.reshape(-1, 1)
if pred.shape[1] != len(targets) or pred.shape[0] != len(y_test):
    raise ValueError(f"Prediction shape {pred.shape} incompatible with targets {targets}")

rows, per = [], {}
for i, t in enumerate(targets):
    yt, yp = y_test[t].to_numpy(), pred[:, i]
    vals = {
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "mae": float(mean_absolute_error(yt, yp)),
        "r2": float(r2_score(yt, yp)),
    }
    per[t] = vals
    for m in ["rmse", "mae", "r2"]:
        rows.append({"model": selected_name, "metric": f"{m}_{t}", "value": vals[m]})

macro = {m: float(np.mean([per[t][m] for t in targets])) for m in ["rmse", "mae", "r2"]}
for m in ["rmse", "mae", "r2"]:
    rows.append({"model": selected_name, "metric": m, "value": macro[m]})

df = pd.DataFrame(rows)
df_save = df.copy()
df_save["value"] = df_save["value"].map(lambda x: float(f"{x:.4g}") if np.isfinite(x) else x)
mlca_runtime.save_table(df_save, "test_metrics.csv", caption="Test metrics for validation-selected candidate")

for r in rows:
    mlca_runtime.record_metric(r["metric"], float(r["value"]), "test")
if str(primary).lower() not in macro:
    raise KeyError(f"Primary metric {primary} was not computed")
mlca_runtime.record_fact_candidate("test_score", float(macro[str(primary).lower()]), source="test_metrics.csv")

print(f"train/test rows: {len(train)}/{len(test)}")
print(f"selected: {selected_name}")
print(f"test {primary}: {macro[str(primary).lower()]:.6g}")
print(f"saved test_metrics.csv rows={len(df_save)}")
