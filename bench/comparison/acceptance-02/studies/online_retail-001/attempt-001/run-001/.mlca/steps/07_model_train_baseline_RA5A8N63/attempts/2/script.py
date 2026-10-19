import mlca_runtime, json, pickle
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

mlca_runtime.set_seed(42)
ins = mlca_runtime.inputs()
train = pd.read_parquet(ins["train"])
proto = json.loads(ins["_protocol"].read_text())
with open(ins["05_preprocess_fit_preprocessing_HR6HVNXD/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)

targets = set(proto.get("targets", []) or [])
excluded = set(proto.get("excluded_columns", []) or [])
group_col = proto.get("group_column")
drop = targets | excluded | ({group_col} if group_col else set())
feature_cols = [c for c in train.columns if c not in drop]
if hasattr(pre, "feature_names_in_"):
    allowed = set(pre.feature_names_in_)
    feature_cols = [c for c in feature_cols if c in allowed]
X = train[feature_cols]
print("train", X.shape, "features", len(feature_cols))

records = []
best = None
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    pipe = Pipeline([("pre", pre), ("km", km)])
    fitted = mlca_runtime.fit_record(pipe, X, "train")
    Xt = fitted[:-1].transform(X)
    Xt = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)
    labels = fitted[-1].labels_
    n = Xt.shape[0]
    rng = np.random.default_rng(42)
    sub = rng.choice(n, min(10000, n), replace=False)
    sil = float(silhouette_score(Xt[sub], labels[sub], random_state=42))
    db = float(davies_bouldin_score(Xt[sub], labels[sub]))
    ch = float(calinski_harabasz_score(Xt[sub], labels[sub]))
    records.append({"k": k, "silhouette": sil, "davies_bouldin": db, "calinski_harabasz": ch})
    if best is None or sil > best[0]:
        best = (sil, fitted, k)

res = pd.DataFrame(records).round(4)
mlca_runtime.save_table(res, "kmeans_candidates.csv", "KMeans candidates")
mlca_runtime.record_metric("silhouette", float(best[0]), "train")
mlca_runtime.record_fact_candidate("n_clusters", int(best[2]), source="kmeans_candidates.csv")
mlca_runtime.save_model(best[1], "model.pkl")
print("best_k", best[2], "silhouette", round(best[0], 4))
