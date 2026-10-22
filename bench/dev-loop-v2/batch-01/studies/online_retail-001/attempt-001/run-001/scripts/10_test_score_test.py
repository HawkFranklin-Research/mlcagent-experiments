import mlca_runtime, json, pickle
import pandas as pd, numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
task = protocol["task"]; primary = task["primary_metric"]
targets = protocol.get("targets", []) or []
excluded = protocol.get("excluded_columns", []) or []
group = protocol.get("group_column")

val_path = inp["09_validate_score_validation_TZGPKJKF/validation_metrics.parquet"]
vm = pd.read_parquet(val_path)
if {"model", "metric", "value"}.issubset(vm.columns):
    sub = vm[vm["metric"] == primary]
    best_model = sub.loc[sub["value"].astype(float).idxmax(), "model"]
elif primary in vm.columns and "model" in vm.columns:
    best_model = vm.loc[vm[primary].astype(float).idxmax(), "model"]
else:
    raise ValueError(f"validation_metrics has unexpected columns: {list(vm.columns)}")

if "alt" in str(best_model).lower():
    model_key = "08_model_train_alternative_RH7MFB6Q/alternative_model.pkl"
else:
    model_key = "07_model_train_baseline_XYVYQTXX/model.pkl"
model = pickle.loads(inp[model_key].read_bytes())

train_df = pd.read_parquet(inp["train"])
test_df = pd.read_parquet(inp["test"])

feat_cols = list(getattr(model, "feature_names_in_", []))
if not feat_cols:
    drop = set(targets) | set(excluded) | ({group} if group else set())
    feat_cols = [c for c in train_df.columns if c not in drop]
    truncated = []
    for c in feat_cols:
        s = train_df[c]
        nn = s.dropna().unique()
        is_id = len(nn) == len(train_df) and (
            pd.api.types.is_integer_dtype(s) or pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s)
        )
        if not is_id:
            truncated.append(c)
    feat_cols = truncated

Xtr = train_df[feat_cols]; Xte = test_df[feat_cols]

def labels_and_metrics(X):
    try:
        labels = model.predict(X)
    except AttributeError:
        labels = model.named_steps[list(model.named_steps)[-1]].labels_ if False else None
        raise
    Xt = model.transform(X) if hasattr(model, "transform") else X
    n = len(Xt)
    if n > 10000:
        s = silhouette_score(Xt, labels, sample_size=10000, random_state=42)
    else:
        s = silhouette_score(Xt, labels)
    return labels, {"silhouette": float(s),
                    "calinski_harabasz": float(calinski_harabasz_score(Xt, labels)),
                    "davies_bouldin": float(davies_bouldin_score(Xt, labels))}

tr_labels, _ = labels_and_metrics(Xtr)
te_labels, te_metrics = labels_and_metrics(Xte)

tm = pd.DataFrame([{"model": str(best_model), "metric": k, "value": round(v, 6)}
                   for k, v in te_metrics.items()])
mlca_runtime.save_table(tm, "test_metrics.csv")

mlca_runtime.record_metric(primary, te_metrics[primary], "test")
mlca_runtime.record_fact_candidate("test_score", te_metrics[primary],
                                   source="test_metrics.csv")

assign = pd.concat([
    pd.DataFrame({"row_id": train_df.index, "partition": "train", "cluster": tr_labels}),
    pd.DataFrame({"row_id": test_df.index,  "partition": "test",  "cluster": te_labels}),
], ignore_index=True)
mlca_runtime.save_row_outputs(assign, "assignments.parquet")

print(f"selected={best_model} primary={primary}={te_metrics[primary]:.4f}")
print(f"train_rows={len(train_df)} test_rows={len(test_df)}")
print(f"metrics={te_metrics}")
