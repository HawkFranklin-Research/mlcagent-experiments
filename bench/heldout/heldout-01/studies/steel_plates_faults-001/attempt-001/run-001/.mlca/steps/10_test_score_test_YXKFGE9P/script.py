import mlca_runtime, json, pickle
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, hamming_loss

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol.get("group_column")
primary = protocol["task"]["primary_metric"]

train = pd.read_parquet(inp["train"])
test = pd.read_parquet(inp["test"])

drop = set(targets) | set(excluded) | ({group} if group else set())
feat_cols = [c for c in train.columns if c not in drop]
for c in list(feat_cols):
    s = train[c]
    if s.nunique(dropna=False) == len(train) and not pd.api.types.is_float_dtype(s) and (
        pd.api.types.is_object_dtype(s) or pd.api.types.is_integer_dtype(s)
    ):
        feat_cols.remove(c)

vm = pd.read_csv(inp["09_validate_score_validation_9MDT3SG5/validation_metrics.csv"])
if {"model", "metric", "value"}.issubset(vm.columns):
    cand = vm[vm["metric"].astype(str).str.lower() == primary.lower()].copy()
elif {"model", primary}.issubset(vm.columns):
    cand = vm[["model", primary]].rename(columns={primary: "value"})
else:
    raise ValueError(f"validation metrics columns: {list(vm.columns)}")
cand = cand.groupby("model", as_index=False)["value"].max()
best_name = cand.loc[cand["value"].idxmax(), "model"]
low = str(best_name).lower()
if "baseline" in low or "07" in low:
    model_path = inp["07_model_train_baseline_KWCD46XN/model.pkl"]
elif "alternative" in low or "08" in low or "latest" in low:
    model_path = inp["08_model_train_alternative_6JQ0NZNF/alternative_model.pkl"]
else:
    model_path = inp["latest_model"]

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

pre = load_pkl(inp["05_preprocess_fit_preprocessing_CSS2QFXR/preprocessor.pkl"])
sel = load_pkl(inp["06_features_retain_features_2A3B8CCD/selector_all.pkl"])
model = load_pkl(model_path)

def get_cols(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        for _, step in obj.steps:
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return None

def transform_front(obj, X):
    cols = get_cols(obj)
    Xo = X[cols] if cols and all(c in X.columns for c in cols) else X
    return obj.transform(Xo)

def predict_labels(model, X):
    if isinstance(model, Pipeline):
        cols = get_cols(model) or feat_cols
        return model.predict(X[list(cols)])
    Xt = X
    if sel is not None:
        if isinstance(sel, Pipeline):
            Xt = transform_front(sel, X)
        else:
            if pre is not None:
                Xt = transform_front(pre, X)
            cols = get_cols(sel)
            if cols and isinstance(Xt, pd.DataFrame) and all(c in Xt.columns for c in cols):
                Xt = Xt[cols]
            Xt = sel.transform(Xt)
    elif pre is not None:
        Xt = transform_front(pre, X)
    return model.predict(Xt)

raw_pred = np.asarray(predict_labels(model, test))
if raw_pred.ndim != 2:
    raise ValueError(f"expected 2D predictions, got shape {raw_pred.shape}")

def encode_col(vals, train_series):
    if pd.api.types.is_numeric_dtype(train_series):
        return np.asarray(vals, dtype=int)
    mapping = {v: i for i, v in enumerate(pd.unique(train_series.dropna()))}
    valid_ints = set(mapping.values())
    out = []
    for v in vals:
        if v in mapping:
            out.append(mapping[v])
        elif isinstance(v, (int, np.integer)) and int(v) in valid_ints:
            out.append(int(v))
        else:
            out.append(-1)
    return np.asarray(out, dtype=int)

y_true = np.column_stack([encode_col(test[t].to_numpy(), train[t]) for t in targets])
y_pred = np.column_stack([encode_col(raw_pred[:, i], train[t]) for i, t in enumerate(targets)])

f1s = [f1_score(y_true[:, i], y_pred[:, i], average="macro", zero_division=0) for i in range(len(targets))]
hls = [hamming_loss(y_true[:, i], y_pred[:, i]) for i in range(len(targets))]
macro_f1 = float(np.mean(f1s))
hamming_avg = float(np.mean(hls))
subset_acc = float(np.mean(np.all(y_true == y_pred, axis=1)))

rows = [
    {"model": best_name, "metric": primary, "value": macro_f1},
    {"model": best_name, "metric": "hamming_loss_avg", "value": hamming_avg},
    {"model": best_name, "metric": "subset_accuracy", "value": subset_acc},
]
for i, t in enumerate(targets):
    rows.append({"model": best_name, "metric": f"macro_f1_{t}", "value": f1s[i]})
    rows.append({"model": best_name, "metric": f"hamming_loss_{t}", "value": hls[i]})
metrics_df = pd.DataFrame(rows)
metrics_df["value"] = metrics_df["value"].map(lambda x: float(f"{x:.4g}"))
mlca_runtime.save_table(metrics_df, "test_metrics.csv", caption="Test metrics")
mlca_runtime.record_metric(primary, macro_f1, "test")
mlca_runtime.record_fact_candidate("test_score", macro_f1, source="test_metrics.csv")

print(f"selected={best_name}")
print(f"test_rows={len(test)} features={len(feat_cols)}")
print(f"{primary}={macro_f1:.4f} hamming_avg={hamming_avg:.4f} subset_acc={subset_acc:.4f}")
