import json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, hamming_loss
from sklearn.pipeline import Pipeline
import mlca_runtime

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
for k in ("targets", "excluded_columns", "group_column", "task"):
    if k not in protocol:
        raise KeyError(f"Missing {k}; available={list(protocol)}")
targets = protocol["targets"]
excluded = set(protocol["excluded_columns"]) | set(targets) | {protocol["group_column"]}
primary = protocol["task"]["primary_metric"]

train = pd.read_parquet(inp["train"])
test = pd.read_parquet(inp["test"])

feature_cols = []
for c in train.columns:
    if c in excluded:
        continue
    s = train[c]
    if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=False) == len(s):
        continue
    if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and s.nunique(dropna=False) == len(s):
        continue
    feature_cols.append(c)

vm = pd.read_csv(inp["10_validate_score_validation_3J1WH09P/validation_metrics.csv"])
if "metric" in vm.columns:
    sub = vm[vm["metric"].astype(str) == primary]
    if sub.empty:
        sub = vm[vm["metric"].astype(str).str.contains(primary, case=False, regex=False)]
    model_col = "model" if "model" in vm.columns else vm.columns[0]
    value_col = "value" if "value" in vm.columns else [c for c in vm.columns if c not in (model_col, "metric")][0]
    best = sub.loc[sub[value_col].astype(float).idxmax()]
    selected_name = str(best[model_col])
else:
    model_col = "model" if "model" in vm.columns else vm.columns[0]
    selected_name = str(vm.loc[vm[primary].astype(float).idxmax(), model_col])

base_key = "08_model_train_baseline_RDZRN5AQ/model.pkl"
alt_key = "09_model_train_alternative_J8YMJGXB/alternative_model.pkl"
low = selected_name.lower()
if "baseline" in low:
    model_key = base_key
elif "alternative" in low or "alt" in low:
    model_key = alt_key
elif "latest" in low:
    model_key = "latest_model"
else:
    model_key = next((k for k in (base_key, alt_key, "latest_model") if low in k.lower()), "latest_model")

def load_pkl(p):
    with open(p, "rb") as f:
        return pickle.load(f)

def unwrap(o):
    if isinstance(o, dict):
        for k in ("model", "estimator", "pipeline"):
            if k in o:
                return o[k]
    return o

pre = unwrap(load_pkl(inp["06_preprocess_fit_preprocessing_DB5T0R19/preprocessor.pkl"]))
sel = unwrap(load_pkl(inp["07_features_retain_features_05E7EJVW/selector_all.pkl"]))
mdl = unwrap(load_pkl(inp[model_key]))

def has_named(o, names):
    return isinstance(o, Pipeline) and any(n in o.named_steps for n in names)

if isinstance(mdl, Pipeline) and len(mdl.named_steps) > 1 and any("preprocess" in n or "select" in n for n in mdl.named_steps):
    complete = mdl
elif has_named(sel, ["preprocessor", "preprocess"]):
    complete = Pipeline(list(sel.steps) + [("model", mdl)])
elif has_named(pre, ["selector", "select"]):
    complete = Pipeline(list(pre.steps) + [("model", mdl)])
else:
    complete = Pipeline([("preprocessor", pre), ("selector", sel), ("model", mdl)])

first = complete.steps[0][1] if isinstance(complete, Pipeline) else complete
raw_cols = getattr(first, "feature_names_in_", None)
raw_cols = list(raw_cols) if raw_cols is not None else feature_cols
X_test = test.loc[:, raw_cols].copy()

if hasattr(complete, "predict"):
    preds = complete.predict(X_test)
else:
    proba = complete.predict_proba(X_test)
    preds = np.column_stack([np.argmax(p, axis=1) for p in proba])
preds = np.asarray(preds)
if preds.ndim == 1:
    preds = preds.reshape(-1, 1)

classes = {t: sorted(pd.unique(train[t].dropna())) for t in targets}
maps = {t: {c: i for i, c in enumerate(classes[t])} for t in targets}
per_f1, per_ham = {}, {}
matches = np.ones(len(test), dtype=bool)
for i, t in enumerate(targets):
    yt = pd.Series(test[t].to_numpy()).reset_index(drop=True)
    yp = pd.Series(preds[:, i]).reset_index(drop=True)
    yp_num = pd.api.types.is_numeric_dtype(yp)
    yt_num = pd.api.types.is_numeric_dtype(yt)
    if yp_num and not yt_num:
        yt_eval = yt.map(maps[t])
        yp_eval = yp.astype(int)
    else:
        yt_eval = yt
        yp_eval = yp
    per_f1[t] = float(f1_score(yt_eval, yp_eval, average="macro", zero_division=0))
    per_ham[t] = float(hamming_loss(yt_eval, yp_eval))
    matches &= (np.asarray(yt_eval) == np.asarray(yp_eval))

macro_f1 = float(np.mean(list(per_f1.values())))
hamming_avg = float(np.mean(list(per_ham.values())))
subset_acc = float(np.mean(matches))

rows = [
    {"model": selected_name, "metric": primary, "value": macro_f1},
    {"model": selected_name, "metric": "hamming_loss", "value": hamming_avg},
    {"model": selected_name, "metric": "subset_accuracy", "value": subset_acc},
]
for t in targets:
    rows.append({"model": selected_name, "metric": f"{t}_macro_f1", "value": per_f1[t]})
    rows.append({"model": selected_name, "metric": f"{t}_hamming_loss", "value": per_ham[t]})
out = pd.DataFrame(rows)
out["value"] = out["value"].map(lambda v: float(f"{float(v):.4g}"))
mlca_runtime.save_table(out, "test_metrics.csv", caption="Test metrics for validation-selected model")
mlca_runtime.record_metric(primary, macro_f1, split="test")
mlca_runtime.record_fact_candidate("test_score", macro_f1, source="test_metrics.csv")
print(f"selected={selected_name}")
print(f"n_test={len(test)} features={len(raw_cols)}")
print(f"{primary}={macro_f1:.4f}")
