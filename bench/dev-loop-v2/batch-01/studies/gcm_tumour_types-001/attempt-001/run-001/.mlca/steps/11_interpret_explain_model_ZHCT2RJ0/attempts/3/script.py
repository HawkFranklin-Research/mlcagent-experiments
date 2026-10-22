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

train = pd.read_parquet(paths["train"])
val = pd.read_parquet(paths["val"])
print("rows train/val", len(train), len(val))

drop = set(targets) | set(excluded)
if group:
    drop.add(group)

def is_keylike(s):
    return s.nunique(dropna=False) == len(s) and s.dtype.kind in "OUSiu"

feats = [c for c in train.columns if c not in drop and not is_keylike(train[c])]

def load_pkl(key):
    with open(paths[key], "rb") as fh:
        return pickle.load(fh)

cand_keys = [
    "latest_model",
    "08_model_train_alternative_R7D6ZBXG/alternative_model.pkl",
    "07_model_train_logistic_FX65AH12/model.pkl",
]
valid = []
for key in cand_keys:
    try:
        m = load_pkl(key)
    except Exception:
        continue
    if not hasattr(m, "predict_proba"):
        continue
    cols = list(getattr(m, "feature_names_in_", []))
    if not cols and hasattr(m, "named_steps"):
        first = next(iter(m.named_steps.values()), None)
        cols = list(getattr(first, "feature_names_in_", []))
    if cols and all(c in val.columns for c in cols):
        valid.append((len(cols), m, cols))
if not valid:
    raise ValueError("No saved model with predict_proba and known validation input columns")

_, model, input_cols = max(valid, key=lambda x: x[0])
bad = [c for c in input_cols if c in drop or c not in feats]
if bad:
    raise ValueError("Model input columns outside declared feature set: %s" % bad[:5])

X = val.loc[:, input_cols].copy()
y = val[targets[0]].to_numpy()
classes = getattr(model, "classes_", None)
if classes is None and hasattr(model, "named_steps"):
    classes = getattr(list(model.named_steps.values())[-1], "classes_", None)

def score(frame):
    p = model.predict_proba(frame)
    if p.ndim == 1:
        p = p.reshape(-1, 1)
    if p.shape[1] == 2 and (classes is None or len(classes) == 2):
        return roc_auc_score(y, p[:, 1])
    if classes is not None:
        return roc_auc_score(y, p, multi_class="ovr", average="macro", labels=classes)
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
