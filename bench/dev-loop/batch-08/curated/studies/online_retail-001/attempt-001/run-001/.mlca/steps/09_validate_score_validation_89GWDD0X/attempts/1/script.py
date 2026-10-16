import mlca_runtime, json, pickle
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())

task = proto["task"]
primary = task["primary_metric"]
targets = proto.get("targets", []) or []
excluded = proto.get("excluded_columns", []) or []
group_col = proto.get("group_column")

def load_pkl(key):
    with open(inputs[key], "rb") as f:
        return pickle.load(f)

def keylike(s):
    if not (pd.api.types.is_integer_dtype(s) or pd.api.types.is_object_dtype(s)
            or pd.api.types.is_string_dtype(s)):
        return False
    return len(s) > 0 and s.nunique(dropna=False) == len(s)

def sig4(x):
    if x is None: return float("nan")
    x = float(x)
    if np.isnan(x): return x
    return float(f"{x:.4g}")

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)
feat_cols = [c for c in train.columns if c not in drop and not keylike(train[c])]

pre = load_pkl("05_preprocess_fit_preprocessing_FN0Z178K/preprocessor.pkl")
sel = load_pkl("06_features_retain_features_YJWZ54M0/selector_all.pkl")

sel_steps = dict(sel.named_steps) if isinstance(sel, Pipeline) else {}
if pre is not None and any(s is pre for s in sel_steps.values()):
    frontend = sel
elif pre is not None:
    frontend = Pipeline([("preprocessor", pre), ("selector", sel)])
else:
    frontend = sel

def cols_for(obj, default):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    return default

fcols = cols_for(frontend, feat_cols)
Xf_val = np.asarray(frontend.transform(val[fcols]))

cands = [("baseline", "07_model_train_baseline_SMYBENGX/model.pkl"),
         ("alternative", "latest_model")]
models, labels_map, feat_map = {}, {}, {}

for name, key in cands:
    m = load_pkl(key)
    models[name] = m
    cols = cols_for(m, feat_cols)
    Xpred = val[cols]
    if isinstance(m, Pipeline) and len(m.steps) >= 2:
        try:
            labels = np.asarray(m.predict(Xpred))
            Xf = np.asarray(m[:-1].transform(Xpred))
        except Exception:
            labels = np.asarray(m.predict(Xf_val))
            Xf = Xf_val
    else:
        labels = np.asarray(m.predict(Xf_val))
        Xf = Xf_val
    labels_map[name] = labels
    feat_map[name] = Xf

def compute(Xf, labels):
    if len(np.unique(labels)) < 2:
        return {"silhouette": float("nan"),
                "calinski_harabasz": float("nan"),
                "davies_bouldin": float("nan")}
    n = len(labels)
    if n > 10000:
        print(f"diagnostic subsample: 10000 of {n}")
        idx = np.random.RandomState(42).choice(n, 10000, replace=False)
        Xs, ls = Xf[idx], labels[idx]
    else:
        Xs, ls = Xf, labels
    return {"silhouette": float(silhouette_score(Xs, ls, random_state=42)),
            "calinski_harabasz": float(calinski_harabasz_score(Xs, ls)),
            "davies_bouldin": float(davies_bouldin_score(Xs, ls))}

rows, per_model = [], {}
for name in labels_map:
    m = compute(feat_map[name], labels_map[name])
    per_model[name] = m
    for k, v in m.items():
        rows.append({"model": name, "metric": k, "value": sig4(v)})

out_df = pd.DataFrame(rows)
mlca_runtime.save_table(out_df, "validation_metrics.csv",
                        caption="Validation clustering metrics per candidate")

higher_better = primary not in ("davies_bouldin",)
primary_scores = {n: per_model[n].get(primary, float("nan")) for n in per_model}
best_name = (max if higher_better else min)(primary_scores, key=primary_scores.get)
best_score = primary_scores[best_name]

mlca_runtime.record_metric(primary, float(best_score), split="val")
mlca_runtime.record_fact_candidate("best_validation_score", float(best_score),
                                   source="validation_metrics.csv")
mlca_runtime.record_fact_candidate("best_model", best_name,
                                   source="validation_metrics.csv")

mlca_runtime.save_model(models[best_name], "frozen_cluster_model")

print(f"features: {len(feat_cols)}  val rows: {len(val)}")
print(f"val {primary}: " + ", ".join(f"{k}={v:.4f}" for k, v in primary_scores.items()))
print(f"frozen: {best_name}  {primary}={best_score:.4f}")
