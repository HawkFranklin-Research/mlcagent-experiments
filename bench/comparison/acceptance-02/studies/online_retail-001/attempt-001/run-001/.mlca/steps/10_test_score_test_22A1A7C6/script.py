import json, pickle
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
out = mlca_runtime.output_dir()
protocol = json.loads(inp["_protocol"].read_text())

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol.get("group_column")
primary = protocol["task"]["primary_metric"]

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
test = pd.read_parquet(inp["test"])

with open(inp["07_model_train_baseline_RA5A8N63/model.pkl"], "rb") as f:
    model = pickle.load(f)
with open(inp["06_features_retain_features_T3RMYHN2/selector_all.pkl"], "rb") as f:
    selector = pickle.load(f)
with open(inp["05_preprocess_fit_preprocessing_HR6HVNXD/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)

feat_cols = [c for c in train.columns if c not in set(targets) | set(excluded) | set([group] if group else [])]
for c in feat_cols:
    if train[c].dtype == object or (pd.api.types.is_integer_dtype(train[c]) and train[c].nunique() == len(train) and len(train) > 20):
        raise ValueError(f"key-like column in features: {c}")

def get_pipeline(obj):
    from sklearn.pipeline import Pipeline
    return obj if isinstance(obj, Pipeline) else None

# Build pipeline frontend: preprocessor -> selector -> clusterer (already inside model if Pipeline)
pipe = get_pipeline(model)
if pipe is None:
    from sklearn.pipeline import Pipeline
    steps = []
    if get_pipeline(pre) is not None:
        steps += list(pre.steps)
    else:
        steps.append(("pre", pre))
    if get_pipeline(selector) is not None:
        steps += list(selector.steps)
    else:
        steps.append(("sel", selector))
    steps.append(("model", model))
    pipe = Pipeline(steps)

X_train = train[feat_cols]
X_test = test[feat_cols]

# Test only scoring
labels_test = pipe.predict(X_test)
X_test_t = pipe[:-1].transform(X_test)
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
n = len(X_test_t)
if n > 10000:
    rng = np.random.RandomState(42)
    idx = rng.choice(n, 10000, replace=False)
    Xs, ls = X_test_t[idx], labels_test[idx]
    print(f"subsample silhouette n={len(idx)}")
else:
    Xs, ls = X_test_t, labels_test

sil = float(silhouette_score(Xs, ls, random_state=42))
ch = float(calinski_harabasz_score(Xs, ls))
db = float(davies_bouldin_score(Xs, ls))

defaults = {"silhouette": sil, "calinski_harabasz": ch, "davies_bouldin": db}
metrics = []
for name in ["silhouette", "calinski_harabasz", "davies_bouldin"]:
    metrics.append({"model": "candidate", "metric": name, "value": round(defaults[name], 6)})
df_metrics = pd.DataFrame(metrics)
mlca_runtime.save_table(df_metrics, "test_metrics.csv", "test clustering metrics")

mlca_runtime.record_metric(primary, float(defaults.get(primary, sil)), "test")

# Assignments for train and test
labels_train = pipe.predict(X_train)
rows = []
rows.append(pd.DataFrame({"row_id": train.index, "partition": "train", "cluster": labels_train}))
rows.append(pd.DataFrame({"row_id": test.index, "partition": "test", "cluster": labels_test}))
assign = pd.concat(rows, ignore_index=True)
mlca_runtime.save_row_outputs(assign, "assignments.parquet")

mlca_runtime.record_fact_candidate("test_score", float(defaults.get(primary, sil)), source="test_metrics.csv")
print(f"primary={primary} value={defaults.get(primary, sil):.4f} n_test={len(test)}")
