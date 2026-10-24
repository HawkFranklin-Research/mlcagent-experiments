import json, pickle
import numpy as np
import pandas as pd
import mlca_runtime
from sklearn.metrics import get_scorer

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())
targets = proto["targets"]
excluded = set(proto["excluded_columns"])
group = proto.get("group_column")
task = proto["task"]
n_classes = task.get("n_classes", 2)
scoring = "roc_auc" if n_classes <= 2 else "roc_auc_ovr"

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])
print(f"train={train.shape} val={val.shape} scoring={scoring}")

model = pickle.load(open(inp["latest_model"], "rb"))
yv = val[targets[0]]

if hasattr(model, "feature_names_in_"):
    feats = list(model.feature_names_in_)
else:
    feats = [c for c in train.columns
             if c not in targets and c not in excluded and c != group]
    feats = [c for c in feats if not (
        len(train[c]) == train[c].nunique()
        and not pd.api.types.is_float_dtype(train[c]))]
print(f"n_features={len(feats)}")

def rank_by_coef(model, feats):
    est = list(model.named_steps.values())[-1] if hasattr(model, "named_steps") else model
    if hasattr(est, "coef_"):
        imp = np.abs(est.coef_)
        imp = imp.mean(axis=0) if imp.shape[0] > 1 else imp.ravel()
    elif hasattr(est, "feature_importances_"):
        imp = np.asarray(est.feature_importances_)
    else:
        return None
    try:
        if hasattr(model, "named_steps") and len(model.named_steps) > 1:
            names = list(model[:-1].get_feature_names_out(list(feats)))
        else:
            names = list(feats)
    except Exception:
        names = list(feats)
    if len(names) != len(imp):
        return None
    names = [str(n).split("__", 1)[-1] for n in names]
    return pd.Series(np.asarray(imp, float), index=names)

Xv = val[feats]
scorer = get_scorer(scoring)

if len(feats) > 500:
    r = rank_by_coef(model, feats)
    if r is None:
        top_feats, n_repeats = [], 0
        method = "no reliable ranking; limitation reported"
    else:
        top_feats = [f for f in r.sort_values(ascending=False).index if f in feats][:50]
        n_repeats = 5
        method = f"coef-rank top{len(top_feats)}, n_repeats=5"
else:
    top_feats, n_repeats = feats, 20
    method = f"full permutation, n_repeats=20"

if top_feats:
    base = scorer(model, Xv, yv)
    rng = np.random.RandomState(42)
    means, stds = [], []
    for col in top_feats:
        orig = Xv[col].values.copy()
        deltas = []
        for _ in range(n_repeats):
            Xv[col] = rng.permutation(orig)
            deltas.append(base - scorer(model, Xv, yv))
        Xv[col] = orig
        means.append(float(np.mean(deltas)))
        stds.append(float(np.std(deltas)))
    res = pd.DataFrame({"feature": top_feats,
                        "importance_mean": means,
                        "importance_sd": stds})
    res = res.sort_values("importance_mean", ascending=False).reset_index(drop=True)
else:
    res = pd.DataFrame({"feature": pd.Series(dtype=str),
                        "importance_mean": pd.Series(dtype=float),
                        "importance_sd": pd.Series(dtype=float)})

head = res.head(15).copy()
for c in ["importance_mean", "importance_sd"]:
    head[c] = head[c].apply(lambda x: float(f"{x:.4g}") if pd.notna(x) else x)

mlca_runtime.save_table(head, "interpretation.csv", caption=method)
if len(head):
    mlca_runtime.record_fact_candidate("top_feature", str(head.iloc[0]["feature"]),
                                       source="interpretation.csv")
print(f"method={method} top={head.iloc[0]['feature'] if len(head) else 'NA'}")
