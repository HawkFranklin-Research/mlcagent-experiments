import json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import mlca_runtime

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()

protocol = json.loads(inp["_protocol"].read_text())
pm = protocol["task"]["primary_metric"]
targets = protocol.get("targets", [])
excluded = protocol.get("excluded_columns", [])
group_col = protocol.get("group_column", None)

def read_pkl(k):
    with open(inp[k], "rb") as f:
        return pickle.load(f)

train = pd.read_parquet(inp["train"])
test = pd.read_parquet(inp["test"])
pre = read_pkl("05_preprocess_fit_preprocessing_WG30AQ6C/preprocessor.pkl")
sel = read_pkl("06_features_retain_features_SSZF8P1C/selector_all.pkl")
baseline = read_pkl("07_model_train_baseline_PYZ8RSVK/model.pkl")
alternative = read_pkl("08_model_train_alternative_3CGWKKCJ/alternative_model.pkl")

# feature columns (RL-0007)
all_cols = list(train.columns)
feature_cols = []
for c in all_cols:
    if c in targets or c in excluded or c == group_col:
        continue
    s = train[c]
    if s.nunique() == len(s):
        if s.dtype == object or (np.issubdtype(s.dtype, np.integer) and s.min() >= 0 and s.max() == len(s) - 1):
            continue
    feature_cols.append(c)

def get_cols(obj):
    return list(obj.feature_names_in_) if hasattr(obj, "feature_names_in_") else feature_cols

def make_runner(model):
    if isinstance(model, Pipeline) and model.named_steps:
        cols = get_cols(model)
        pre_steps = model[:-1]
        def run(X):
            Xs = X[cols]
            Xt = pre_steps.transform(Xs) if len(pre_steps.steps) > 0 else Xs
            labels = model.predict(Xs)
            return Xt, np.asarray(labels)
        return run
    pcols = get_cols(pre) if pre is not None and hasattr(pre, "feature_names_in_") else feature_cols
    def run(X):
        Xs = X[pcols]
        if pre is not None:
            Xs = pre.transform(Xs)
        if sel is not None:
            Xs = sel.transform(Xs)
        labels = model.predict(Xs) if hasattr(model, "predict") else model.fit_predict(Xs)
        return Xs, np.asarray(labels)
    return run

runners = {"baseline": make_runner(baseline), "alternative": make_runner(alternative)}

# select best candidate on validation
vm = pd.read_csv(inp["09_validate_score_validation_RBGGTR3K/validation_metrics.csv"])
if "split" in vm.columns:
    vm = vm[vm["split"].isin(["val", "validation"])]
sub = vm[vm["metric"] == pm]
greater = True
if pm in ("davies_bouldin", "inertia"):
    greater = False
if "greater_is_better" in protocol["task"]:
    greater = protocol["task"]["greater_is_better"]
elif "direction" in protocol["task"]:
    greater = protocol["task"]["direction"] == "maximize"
idx = sub["value"].idxmax() if greater else sub["value"].idxmin()
best_name = str(sub.loc[idx, "model"]).lower()
if "baseline" in best_name:
    selected_name, selected_model = "baseline", baseline
else:
    selected_name, selected_model = "alternative", alternative
runner_sel = make_runner(selected_model)

# test metrics for both candidates
funcs = {
    "silhouette": lambda X, y: silhouette_score(X, y, sample_size=min(10000, len(y)), random_state=42),
    "calinski_harabasz": calinski_harabasz_score,
    "davies_bouldin": davies_bouldin_score,
}
rows = []
for name, runner in runners.items():
    Xt, labels = runner(test)
    for mname, fn in funcs.items():
        try:
            v = float(fn(Xt, labels))
        except Exception:
            v = np.nan
        rows.append({"model": name, "metric": mname, "value": v})
metrics_df = pd.DataFrame(rows)
mlca_runtime.save_table(metrics_df, "test_metrics.csv")

# record selected test score
sel_row = metrics_df[(metrics_df["model"] == selected_name) & (metrics_df["metric"] == pm)]
test_score = float(sel_row["value"].iloc[0]) if len(sel_row) else np.nan
mlca_runtime.record_metric(pm, test_score, "test")
mlca_runtime.record_fact_candidate("test_score", test_score, source="test_metrics.csv")

# assignments for the selected model
_, labels_train = runner_sel(train)
_, labels_test = runner_sel(test)
assignments = pd.concat([
    pd.DataFrame({"row_id": train.index, "partition": "train", "cluster": labels_train}),
    pd.DataFrame({"row_id": test.index, "partition": "test", "cluster": labels_test}),
], ignore_index=True)
mlca_runtime.save_row_outputs(assignments, "assignments.parquet")

print(f"test_score={test_score:.4f}")
print(f"metrics_rows={len(metrics_df)} assignments_rows={len(assignments)}")
