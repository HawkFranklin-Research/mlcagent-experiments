import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())
task = proto["task"]
primary = task["primary_metric"]
targets = proto.get("targets", [])
excluded = proto.get("excluded_columns", [])
if "group_column" not in proto:
    raise KeyError(f"group_column missing; top-level keys: {sorted(proto)}")
group = proto["group_column"]

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])

def key_like(s):
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return True
    return pd.api.types.is_integer_dtype(s) and s.notna().all() and s.nunique() == len(s)

drop = set(targets) | set(excluded)
if group is not None:
    drop.add(group)
feature_cols = [c for c in train.columns if c not in drop and not key_like(train[c])]
if not feature_cols:
    raise ValueError("No feature columns found")

def higher_better(metric):
    return "davies" not in metric.lower() and metric.lower() != "db"

def model_cols(model, df):
    cols = getattr(model, "feature_names_in_", None)
    if cols is None:
        return feature_cols
    return [c for c in list(cols) if c in feature_cols and c in df.columns]

def predict_labels(model, df):
    X = df[model_cols(model, df)]
    if hasattr(model, "predict"):
        return np.asarray(model.predict(X))
    if hasattr(model, "labels_") and len(model.labels_) == len(df):
        return np.asarray(model.labels_)
    raise ValueError("selected model cannot assign clusters")

pre = pickle.load(open(inp["05_preprocess_fit_preprocessing_1E8V578C/preprocessor.pkl"], "rb"))
sel = pickle.load(open(inp["06_features_retain_features_Y7PF8JJ2/selector_all.pkl"], "rb"))

def transform_for_metric(model, df):
    cols = model_cols(model, df)
    if isinstance(model, Pipeline):
        front = model[:-1]
        return front.transform(df[cols]) if len(front.steps) else df[cols]
    sel_cols = getattr(sel, "feature_names_in_", None)
    if sel_cols is not None and set(sel_cols).issubset(df.columns):
        return sel.transform(df[list(sel_cols)])
    pre_cols = list(getattr(pre, "feature_names_in_", cols))
    X = pre.transform(df[pre_cols])
    try:
        names = pre.get_feature_names_out()
    except Exception:
        names = None
    if names is not None and not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=names)
    return sel.transform(X)

def primary_value(X, labels):
    n = len(labels)
    kwargs = {"sample_size": min(10000, n), "random_state": 42}
    low = primary.lower()
    if "silhouette" in low:
        return silhouette_score(X, labels, **kwargs)
    if "calinski" in low:
        return calinski_harabasz_score(X, labels)
    if "davies" in low:
        return davies_bouldin_score(X, labels)
    raise ValueError(f"Unsupported primary metric: {primary}")

def try_load(name):
    low = name.lower()
    paths = [
        ("baseline", inp["07_model_train_baseline_SWTMFPFT/model.pkl"]),
        ("alternative", inp["08_model_train_alternative_XQ00QFTN/alternative_model.pkl"]),
        ("latest", inp["latest_model"]),
    ]
    for tag, path in paths:
        if tag in low:
            with open(path, "rb") as f:
                return pickle.load(f)
    return None

def load_all_models():
    paths = [
        ("model.pkl", inp["07_model_train_baseline_SWTMFPFT/model.pkl"]),
        ("alternative_model.pkl", inp["08_model_train_alternative_XQ00QFTN/alternative_model.pkl"]),
        ("latest_model", inp["latest_model"]),
    ]
    loaded = []
    for name, path in paths:
        try:
            with open(path, "rb") as f:
                loaded.append((name, pickle.load(f)))
        except Exception:
            pass
    return loaded

selected_name = None
selected_model = None
vm = pd.read_csv(inp["09_validate_score_validation_H8CSBPBR/validation_metrics.csv"])
cols = {str(c).lower(): c for c in vm.columns}
model_col = cols.get("model") or cols.get("candidate") or cols.get("name")
metric_col = cols.get("metric")
value_col = cols.get("value") or cols.get("score")
if model_col and metric_col and value_col:
    if "split" in cols:
        vm = vm[vm[cols["split"]].astype(str).str.lower().isin(["val", "validation"])]
    rows = vm[vm[metric_col].astype(str).str.lower() == primary.lower()].copy()
    rows["_v"] = pd.to_numeric(rows[value_col], errors="coerce")
    rows = rows[rows["_v"].notna()]
    if len(rows):
        idx = rows["_v"].idxmax() if higher_better(primary) else rows["_v"].idxmin()
        selected_name = str(rows.loc[idx, model_col])
        selected_model = try_load(selected_name)

if selected_model is None:
    scored = []
    for name, model in load_all_models():
        try:
            labels = predict_labels(model, val)
            X = transform_for_metric(model, val)
            scored.append((primary_value(X, labels), name, model))
        except Exception:
            pass
    if not scored:
        raise RuntimeError("No savable candidate model could be scored on validation")
    best = max(scored, key=lambda t: t[0]) if higher_better(primary) else min(scored, key=lambda t: t[0])
    _, selected_name, selected_model = best

test = pd.read_parquet(inp["test"])
test_labels = predict_labels(selected_model, test)
X_test = transform_for_metric(selected_model, test)
if hasattr(X_test, "toarray"):
    X_test = X_test.toarray()
n = len(test_labels)
sil = silhouette_score(X_test, test_labels, sample_size=min(10000, n), random_state=42)
ch = calinski_harabasz_score(X_test, test_labels)
db = davies_bouldin_score(X_test, test_labels)
low = primary.lower()
if "silhouette" in low:
    pv = sil
elif "calinski" in low:
    pv = ch
elif "davies" in low:
    pv = db
else:
    raise ValueError(f"Unsupported primary metric: {primary}")

def s4(x):
    return float(f"{x:.4g}")

rows = [{"model": selected_name, "metric": primary, "value": s4(pv)}]
for name, value in [("silhouette", sil), ("calinski_harabasz", ch), ("davies_bouldin", db)]:
    if name.lower().replace("_", "-") != primary.lower().replace("_", "-"):
        rows.append({"model": selected_name, "metric": name, "value": s4(value)})
metrics_df = pd.DataFrame(rows)
mlca_runtime.save_table(metrics_df, "test_metrics.csv", caption="Test clustering metrics")
mlca_runtime.record_metric(primary, float(pv), split="test")
mlca_runtime.record_fact_candidate("test_score", float(pv), source="test_metrics.csv")

def assignment_frame(model, df, partition):
    return pd.DataFrame({"row_id": df.index, "partition": partition, "cluster": predict_labels(model, df)})

assignments = pd.concat(
    [assignment_frame(selected_model, train, "train"), assignment_frame(selected_model, test, "test")],
    ignore_index=True,
)
mlca_runtime.save_row_outputs(assignments, "assignments.parquet")

print("features", len(feature_cols))
print("selected", selected_name)
print("test", primary, round(float(pv), 4))
print("assignments", assignments.shape)
