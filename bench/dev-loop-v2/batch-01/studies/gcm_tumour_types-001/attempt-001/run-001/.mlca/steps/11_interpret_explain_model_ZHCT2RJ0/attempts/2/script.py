import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

seed = 42
mlca_runtime.set_seed(seed)
paths = mlca_runtime.inputs()
proto = json.loads(paths["_protocol"].read_text())

targets = proto["targets"]
excluded = proto.get("excluded_columns", [])
group = proto.get("group_column")
task = proto["task"]

train = pd.read_parquet(paths["train"])
val = pd.read_parquet(paths["val"])
print("rows train/val", len(train), len(val))

drop = set(targets) | set(excluded)
if group:
    drop.add(group)

def is_keylike(s):
    return s.nunique(dropna=False) == len(s) and (
        s.dtype.kind in "OUS" or s.dtype.kind in "iu"
    )

feats = [c for c in train.columns if c not in drop and not is_keylike(train[c])]

def load_pkl(key):
    with open(paths[key], "rb") as fh:
        return pickle.load(fh)

model = load_pkl("07_model_train_logistic_FX65AH12/model.pkl")
if not hasattr(model, "predict_proba"):
    model = load_pkl("latest_model")

input_cols = list(getattr(model, "feature_names_in_", []))
if not input_cols:
    input_cols = feats
missing = [c for c in input_cols if c not in val.columns]
if missing:
    raise ValueError("Model input columns missing from validation table")
input_cols = [c for c in input_cols if c in feats]
if not input_cols:
    raise ValueError("Model input columns do not overlap declared features")

target = targets[0]
y = val[target].to_numpy()
X = val[input_cols]

classes = list(getattr(model, "classes_", []))

def score(frame):
    p = model.predict_proba(frame)
    if p.ndim == 2 and p.shape[1] == 2 and len(classes) <= 2:
        return roc_auc_score(y, p[:, 1])
    if classes:
        order = np.argsort(np.asarray(classes))
        return roc_auc_score(
            y,
            p[:, order],
            multi_class="ovr",
            average="macro",
            labels=np.asarray(classes)[order],
        )
    return roc_auc_score(y, p, multi_class="ovr", average="macro")

imp_feats = input_cols
for step in getattr(model, "named_steps", {}).values():
    if not hasattr(step, "get_support"):
        continue
    try:
        support = np.asarray(step.get_support())
    except Exception:
        continue
    names = list(getattr(step, "feature_names_in_", []))
    if len(names) == len(support):
        chosen = [n for n, keep in zip(names, support) if keep]
    elif len(support) == len(input_cols):
        chosen = [input_cols[i] for i, keep in enumerate(support) if keep]
    else:
        continue
    if chosen and set(chosen).issubset(set(input_cols)):
        imp_feats = chosen
        break

print("model inputs", len(input_cols), "importance features", len(imp_feats))

base = score(X)
rng = np.random.RandomState(seed)
rows = []
for feature in imp_feats:
    diffs = []
    for _ in range(20):
        Xp = X.copy()
        Xp[feature] = rng.permutation(Xp[feature].to_numpy())
        diffs.append(base - score(Xp))
    rows.append((feature, float(np.mean(diffs)), float(np.std(diffs))))

out = pd.DataFrame(rows, columns=["feature", "importance_mean", "importance_sd"])
out = out.sort_values("importance_mean", ascending=False).head(15).reset_index(drop=True)
for col in ["importance_mean", "importance_sd"]:
    out[col] = out[col].map(lambda v: float(f"{v:.4g}") if np.isfinite(v) else np.nan)

mlca_runtime.save_table(out, "interpretation.csv", caption="Permutation importance")
top = out.loc[0, "feature"] if len(out) else None
mlca_runtime.record_fact_candidate("top_feature", top, source="interpretation.csv")
print("top_feature", top)
