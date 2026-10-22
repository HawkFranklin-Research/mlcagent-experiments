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

def need(key):
    if key not in proto:
        raise KeyError(f"Missing {key}; available top-level keys: {list(proto.keys())}")
    return proto[key]

targets = need("targets")
excluded = need("excluded_columns")
group = need("group_column")
task = need("task")

train = pd.read_parquet(paths["train"])
val = pd.read_parquet(paths["val"])
print("rows train/val", len(train), len(val))

drop = set(targets) | set(excluded)
if group:
    drop.add(group)

def is_keylike(s):
    if s.nunique(dropna=False) != len(s):
        return False
    return s.dtype.kind in "OUS" or s.dtype.kind in "iu"

feats = [c for c in train.columns if c not in drop]
feats = [c for c in feats if not is_keylike(train[c])]

def load_pkl(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)

cand_keys = [
    "latest_model",
    "07_model_train_logistic_FX65AH12/model.pkl",
    "08_model_train_alternative_R7D6ZBXG/alternative_model.pkl",
]
cands = []
for key in cand_keys:
    if key in paths:
        try:
            cands.append((key, load_pkl(paths[key])))
        except Exception:
            pass

def input_names(obj):
    return list(getattr(obj, "feature_names_in_", []))

best = None
best_score = (-1, -1)
for key, obj in cands:
    if not hasattr(obj, "predict_proba"):
        continue
    names = input_names(obj)
    if not names:
        continue
    exact = len(names) == len(feats) and set(names) == set(feats)
    score = (1 if exact else 0, len(names))
    if score > best_score:
        best_score = score
        best = (key, obj, names)

if best is None:
    for key, obj in cands:
        if hasattr(obj, "predict_proba"):
            best = (key, obj, input_names(obj) or feats)
            break
if best is None:
    raise RuntimeError("No saved predictive model with predict_proba found")

model_key, model, input_cols = best
if not set(input_cols).issubset(set(feats)):
    raise ValueError("Model expects columns outside the declared feature set")
print("model", model_key, "input_features", len(input_cols))

target = targets[0]
y = val[target]
X = val[input_cols]

def importance_features():
    steps = getattr(model, "steps", None)
    if not steps:
        return list(input_cols)
    for _, step in steps:
        if hasattr(step, "get_support") and hasattr(step, "feature_names_in_"):
            names = list(step.feature_names_in_)
            try:
                support = step.get_support()
            except Exception:
                continue
            chosen = [n for n, keep in zip(names, support) if keep]
            if chosen and set(chosen).issubset(set(input_cols)):
                return chosen
    return list(input_cols)

imp_feats = importance_features()
print("importance_features", len(imp_feats))

n_classes = task.get("n_classes")
if n_classes is None:
    n_classes = len(np.unique(y))

def score(frame):
    proba = model.predict_proba(frame)
    if n_classes > 2:
        return roc_auc_score(y, proba, multi_class="ovr", average="macro")
    return roc_auc_score(y, proba[:, 1] if proba.ndim == 2 and proba.shape[1] == 2 else proba)

base = score(X)
rng = np.random.RandomState(seed)
rows = []
for feature in imp_feats:
    diffs = []
    for _ in range(20):
        Xp = X.copy(deep=False)
        Xp[feature] = rng.permutation(Xp[feature].to_numpy())
        diffs.append(base - score(Xp))
    rows.append((feature, float(np.mean(diffs)), float(np.std(diffs))))

out = pd.DataFrame(rows, columns=["feature", "importance_mean", "importance_sd"])
out = out.sort_values("importance_mean", ascending=False).head(15).reset_index(drop=True)

def sig4(value):
    return float(f"{value:.4g}") if np.isfinite(value) else np.nan

for col in ["importance_mean", "importance_sd"]:
    out[col] = out[col].map(sig4)

mlca_runtime.save_table(out, "interpretation.csv", caption="Permutation importance")
top = out.loc[0, "feature"] if len(out) else None
mlca_runtime.record_fact_candidate("top_feature", top, source="interpretation.csv")
print("top_feature", top)
