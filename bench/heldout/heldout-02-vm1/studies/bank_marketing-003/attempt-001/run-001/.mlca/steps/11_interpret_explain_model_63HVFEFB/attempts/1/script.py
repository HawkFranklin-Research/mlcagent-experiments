import json
import pickle

import mlca_runtime
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
excluded = set(protocol["excluded_columns"])
group_col = protocol.get("group_column")
task = protocol["task"]
metric = task["primary_metric"]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
print("rows", len(train), len(val))


def is_keylike(s):
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return len(s) > 0 and s.nunique(dropna=False) == len(s)
    if pd.api.types.is_integer_dtype(s):
        return len(s) > 0 and s.nunique(dropna=False) == len(s) and any(
            token in s.name.lower() for token in ("id", "key", "code", "index")
        )
    return False


all_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group_col]
feature_cols = [c for c in all_cols if not is_keylike(train[c])]

with open(inputs["latest_model"], "rb") as f:
    model = pickle.load(f)

model_features = list(getattr(model, "feature_names_in_", feature_cols))
forbidden = set(targets) | excluded | ({group_col} if group_col else set())
if not set(model_features).isdisjoint(forbidden):
    raise ValueError("model features overlap targets/excluded/group columns")

target = targets[0]
y = val[target]
classes = getattr(model, "classes_", None)
if classes is None:
    classes = np.unique(y)
classes = list(classes)

if len(classes) == 2:
    pos = classes[1]
    uniq = set(pd.unique(y))
    if pos not in uniq:
        if pos == 1 and "yes" in uniq:
            pos = "yes"
        elif pos == 0 and "no" in uniq:
            pos = "no"
        elif pos == 1 and True in uniq:
            pos = True
        else:
            pos = sorted(uniq)[-1]
    yy = (y == pos).astype(int)
else:
    yy = y


def score(X):
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
    else:
        p = model.decision_function(X)
    if len(classes) == 2:
        s = p[:, 1] if np.ndim(p) == 2 else p
        return roc_auc_score(yy, s)
    return roc_auc_score(yy, p, multi_class="ovr", average="macro")


X = val[model_features]
base = float(score(X))
n_features = len(model_features)
n_repeats = 20 if n_features <= 500 else 5

if n_features > 500:
    est = model.steps[-1][1] if hasattr(model, "steps") else model
    vals = None
    if hasattr(est, "coef_"):
        vals = np.abs(np.ravel(est.coef_))
    elif hasattr(est, "feature_importances_"):
        vals = np.abs(est.feature_importances_)
    if vals is None or len(vals) != n_features:
        df = pd.DataFrame(columns=["feature", "importance_mean", "importance_sd"])
        mlca_runtime.save_table(df, "interpretation.csv", "Feature ranking unavailable")
        mlca_runtime.record_fact_candidate("top_feature", "unavailable", source="interpretation.csv")
        print("No reliable feature ranking")
        raise SystemExit(0)
    order = np.argsort(vals)[::-1][:50]
    perm_features = [model_features[i] for i in order]
else:
    perm_features = model_features

rng = np.random.default_rng(42)
rows = []
for col in perm_features:
    drops = []
    for _ in range(n_repeats):
        Xp = X.copy()
        Xp[col] = rng.permutation(X[col].to_numpy())
        drops.append(base - score(Xp))
    rows.append({
        "feature": col,
        "importance_mean": float(np.mean(drops)),
        "importance_sd": float(np.std(drops)),
    })

df = pd.DataFrame(rows).sort_values("importance_mean", ascending=False).head(15)
for c in ["importance_mean", "importance_sd"]:
    df[c] = df[c].map(lambda v: float(f"{v:.4g}"))

mlca_runtime.save_table(df, "interpretation.csv", "Permutation importance of frozen model features")
top = df.iloc[0]["feature"] if len(df) else "unavailable"
mlca_runtime.record_fact_candidate("top_feature", top, source="interpretation.csv")
mlca_runtime.record_metric("interpretation_n_features", float(n_features), "val")
mlca_runtime.record_metric("interpretation_n_repeats", float(n_repeats), "val")
print("baseline_auc", round(base, 4))
print("top_feature", top)
print("n_features", n_features, "n_repeats", n_repeats)
print("saved interpretation.csv", df.shape)
