import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
for k in ("targets", "excluded_columns", "group_column", "task"):
    if k not in protocol:
        raise ValueError(f"Missing {k}. Available top-level keys: {list(protocol)}")

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group_col = protocol["group_column"]
primary_name = protocol["task"]["primary_metric"]

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])

drop = set(targets) | set(excluded) | ({group_col} if group_col else set())
feature_cols = []
for c in train.columns:
    if c in drop:
        continue
    s = train[c]
    if (s.dtype == object or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype)) and s.nunique(dropna=False) == len(s):
        continue
    if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=False) == len(s):
        if any(tok in c.lower() for tok in ("id", "code", "key", "index")):
            continue
    feature_cols.append(c)

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

pre = load_pkl(inp["05_preprocess_fit_preprocessing_CSS2QFXR/preprocessor.pkl"])
sel = load_pkl(inp["06_features_retain_features_2A3B8CCD/selector_all.pkl"])
models = [
    ("baseline", load_pkl(inp["07_model_train_baseline_KWCD46XN/model.pkl"])),
    ("alternative", load_pkl(inp["08_model_train_alternative_6JQ0NZNF/alternative_model.pkl"])),
]

def input_names(obj):
    if obj is None:
        return None
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        for step in obj.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return None

def output_names(obj, names, n):
    if hasattr(obj, "get_feature_names_out"):
        try:
            return list(obj.get_feature_names_out(names))
        except Exception:
            try:
                return list(obj.get_feature_names_out())
            except Exception:
                pass
    return [f"f{i}" for i in range(n)]

def as_frame(x, index, cols):
    if hasattr(x, "toarray"):
        x = x.toarray()
    if isinstance(x, pd.DataFrame):
        return x
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if cols is None or len(cols) != x.shape[1]:
        cols = [f"f{i}" for i in range(x.shape[1])]
    return pd.DataFrame(x, index=index, columns=cols)

def selected_frame(raw, raw_cols, model_cols):
    if sel is None:
        if pre is None:
            return raw[raw_cols]
        pre_in = input_names(pre)
        xin = raw[pre_in] if pre_in and all(c in raw.columns for c in pre_in) else raw[raw_cols]
        xp = pre.transform(xin)
        pcols = output_names(pre, list(xin.columns), xp.shape[1])
        return as_frame(xp, raw.index, model_cols if model_cols else pcols)

    sel_in = input_names(sel)
    if sel_in is None or set(sel_in) == set(raw_cols):
        xin = raw[raw_cols] if sel_in is None else raw[sel_in]
        xs = sel.transform(xin)
        cols = model_cols if model_cols else output_names(sel, list(xin.columns), xs.shape[1])
        return as_frame(xs, raw.index, cols)

    if pre is None:
        raise ValueError("Selector expects non-raw columns but no preprocessor is available")
    pre_in = input_names(pre)
    xin = raw[pre_in] if pre_in and all(c in raw.columns for c in pre_in) else raw[raw_cols]
    xp = pre.transform(xin)
    pcols = output_names(pre, list(xin.columns), xp.shape[1])
    xs = sel.transform(as_frame(xp, raw.index, pcols))
    cols = model_cols if model_cols else output_names(sel, pcols, xs.shape[1])
    return as_frame(xs, raw.index, cols)

def model_frame(model, raw, raw_cols):
    need = input_names(model)
    if need is not None and set(need) == set(raw_cols) and all(c in raw.columns for c in need):
        return raw[need]
    if isinstance(model, Pipeline) and need is None:
        return raw[raw_cols]
    x = selected_frame(raw, raw_cols, need)
    if need is not None:
        if all(c in x.columns for c in need):
            return x[need]
        if x.shape[1] == len(need):
            x = x.copy()
            x.columns = need
            return x
        raise ValueError(f"Model expects {len(need)} columns, got {x.shape[1]}")
    return x

def to_numeric_if_possible(a):
    if a.dtype == bool:
        return a.astype(int)
    if a.dtype == object:
        try:
            return a.astype(int)
        except Exception:
            return a
    return a

y_true = to_numeric_if_possible(val[targets].to_numpy())
rows, scores = [], {}

for name, model in models:
    x_model = model_frame(model, val, feature_cols)
    pred = model.predict(x_model)
    if isinstance(pred, list):
        pred = np.column_stack(pred)
    pred = np.asarray(pred)
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
    pred = to_numeric_if_possible(pred)

    f1s, hls = [], []
    for j in range(len(targets)):
        yt, yp = y_true[:, j], pred[:, j]
        if yt.dtype != yp.dtype:
            try:
                yt = yt.astype(int)
                yp = yp.astype(int)
            except Exception:
                pass
        f1s.append(f1_score(yt, yp, average="macro", zero_division=0))
        hls.append(float(np.mean(yt != yp)))

    f1_avg = float(np.mean(f1s))
    hl_avg = float(np.mean(hls))
    subset = float(np.mean(np.all(y_true == pred, axis=1)))
    if "hamming" in primary_name.lower():
        primary_val = hl_avg
    elif "subset" in primary_name.lower() or "accuracy" in primary_name.lower():
        primary_val = subset
    else:
        primary_val = f1_avg

    scores[name] = primary_val
    metric_vals = {primary_name: primary_val, "hamming_loss": hl_avg, "subset_accuracy": subset}
    for metric, value in metric_vals.items():
        rows.append({"model": name, "metric": metric, "value": value})

lower_better = any(k in primary_name.lower() for k in ("loss", "error", "hamming"))
best_name = min(scores, key=scores.get) if lower_better else max(scores, key=scores.get)
best_score = float(scores[best_name])

table = pd.DataFrame(rows, columns=["model", "metric", "value"])
table["value"] = table["value"].map(lambda v: float(f"{v:.4g}"))
mlca_runtime.save_table(table, "validation_metrics.csv", caption="Validation metrics for candidate models")
mlca_runtime.record_metric(primary_name, best_score, "val")
mlca_runtime.record_fact_candidate("best_validation_score", best_score, source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_model", best_name, source="validation_metrics.csv")

print(f"train={train.shape} val={val.shape} features={len(feature_cols)}")
print(f"best={best_name} {primary_name}={best_score:.4g}")
