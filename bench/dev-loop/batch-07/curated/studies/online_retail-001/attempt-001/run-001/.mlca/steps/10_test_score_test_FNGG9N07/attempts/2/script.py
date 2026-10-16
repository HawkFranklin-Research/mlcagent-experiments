import mlca_runtime, json, pickle
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())
targets = proto.get("targets") or []
excluded = proto.get("excluded_columns") or []
group_col = proto.get("group_column")
task = proto["task"]
primary = task["primary_metric"]

train = pd.read_parquet(inputs["train"])
test = pd.read_parquet(inputs["test"])

drop = set(targets) | set(excluded) | ({group_col} if group_col else set())
shared = [c for c in test.columns if c in train.columns and c not in drop]

def is_key(s):
    if not pd.api.types.is_integer_dtype(train[s]):
        return False
    return train[s].is_unique and train[s].nunique() == len(train)

feat_cols = [c for c in shared if not is_key(c)]
X_train = train[feat_cols]
X_test = test[feat_cols]

vm = pd.read_csv(inputs["09_validate_score_validation_T9RQ4KJY/validation_metrics.csv"])
metric_col = "metric" if "metric" in vm.columns else vm.columns[1]
model_col = "model" if "model" in vm.columns else vm.columns[0]
val = vm[vm[metric_col].astype(str) == primary]
if val.empty:
    val = vm
higher = primary.lower() in ("silhouette", "calinski_harabasz", "chi", "adjusted_rand")
best_row = val.sort_values("value", ascending=not higher).iloc[0]
best_name = str(best_row[model_col])

candidates = {
    "baseline": inputs["07_model_train_baseline_7R0X4ZSD/model.pkl"],
    "alternative": inputs["08_model_train_alternative_3C6W5R97/alternative_model.pkl"],
}
model_path = inputs["latest_model"]
for key, p in candidates.items():
    if key in best_name.lower():
        model_path = p
        break
with open(model_path, "rb") as f:
    model = pickle.load(f)

def split_pipe(m):
    if isinstance(m, Pipeline) and len(m.steps) >= 2:
        return Pipeline(m.steps[:-1]), m.steps[-1][1]
    return None, m

prep, clu = split_pipe(model)

def feats(X):
    if prep is not None:
        try:
            return prep.transform(X)
        except Exception:
            return X
    return X

def labels(Xt):
    if hasattr(clu, "predict"):
        return np.asarray(clu.predict(Xt))
    return np.asarray(clu.labels_)

Ft_train, Ft_test = feats(X_train), feats(X_test)
lab_train = labels(Ft_train)
lab_test = labels(Ft_test)

F_arr = np.asarray(Ft_test)
ss = min(10000, len(F_arr))
sil = float(silhouette_score(F_arr, lab_test, sample_size=ss, random_state=42))
ch = float(calinski_harabasz_score(F_arr, lab_test))
db = float(davies_bouldin_score(F_arr, lab_test))

primary_value = {"silhouette": sil, "calinski_harabasz": ch, "davies_bouldin": db}.get(primary.lower())

tm = pd.DataFrame({
    "model": [best_name] * 3,
    "metric": ["silhouette", "calinski_harabasz", "davies_bouldin"],
    "value": [round(sil, 4), round(ch, 4), round(db, 4)],
})
mlca_runtime.save_table(tm, "test_metrics.csv", caption="Test cluster metrics for validation-selected candidate")

if primary_value is None:
    primary_value = sil
mlca_runtime.record_metric(primary, float(primary_value), "test")
mlca_runtime.record_fact_candidate("test_score", float(primary_value), source="test_metrics.csv")

assign = pd.concat([
    pd.DataFrame({"row_id": train.index, "partition": "train", "cluster": lab_train}),
    pd.DataFrame({"row_id": test.index, "partition": "test", "cluster": lab_test}),
], ignore_index=True)
assign["row_id"] = assign["row_id"].astype(str)
mlca_runtime.save_row_outputs(assign, "assignments.parquet")

print(f"model={best_name} test_n={len(F_arr)}")
print(f"{primary}={primary_value:.4f} sil={sil:.4f} ch={ch:.4f} db={db:.4f}")
