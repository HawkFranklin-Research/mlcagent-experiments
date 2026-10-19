import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())
targets = list(protocol["targets"])
excluded = list(protocol.get("excluded_columns", []))
group = protocol.get("group_column")
train = pd.read_parquet(paths["train"])

with open(paths["05_preprocess_fit_preprocessing_8FQ40592/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(paths["06_features_select_features_BD9SGMPY/selector.pkl"], "rb") as f:
    sel = pickle.load(f)


def input_cols(obj):
    fi = getattr(obj, "feature_names_in_", None)
    return None if fi is None else [str(c) for c in fi]


def contains_preprocessor(obj):
    if not isinstance(obj, Pipeline):
        return False
    keys = set(obj.named_steps)
    if keys & {"preprocessor", "preprocessing", "preprocess"}:
        return True
    first = list(obj.named_steps.values())[0]
    return hasattr(first, "transformers")


def feature_columns(df, pre_obj):
    forbidden = set(targets) | set(excluded)
    if group:
        forbidden.add(group)
    cand = [c for c in df.columns if c not in forbidden]
    n = len(df)
    drop = []
    for c in cand:
        s = df[c]
        if pd.api.types.is_string_dtype(s.dtype) or s.dtype == object:
            if s.nunique(dropna=True) == n:
                drop.append(c)
        elif pd.api.types.is_integer_dtype(s.dtype):
            if s.nunique(dropna=True) == n:
                vals = np.sort(s.dropna().unique())
                if len(vals) > 1 and (vals.max() - vals.min() + 1) == n:
                    drop.append(c)
    feat = [c for c in cand if c not in drop]
    pc = input_cols(pre_obj)
    if pc is not None and all(c in df.columns and c not in forbidden for c in pc):
        feat = pc
    return feat


feature_cols = feature_columns(train, pre)
target = targets[0]
if target not in train.columns:
    raise KeyError(f"target {target!r} missing; columns={list(train.columns)}")
if not feature_cols:
    raise ValueError("no feature columns remain after exclusions")

X = train[feature_cols].copy()
y = train[target].copy()

sel_inputs = input_cols(sel)
sel_accepts_raw = (
    sel_inputs is not None
    and len(sel_inputs) == len(feature_cols)
    and set(sel_inputs) == set(feature_cols)
)

if sel_accepts_raw or contains_preprocessor(sel):
    steps = [("selector", sel)]
else:
    steps = [("preprocessor", pre), ("selector", sel)]
steps.append(("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)))
model = Pipeline(steps)

fitted = mlca_runtime.fit_record(model, X, "train", y=y)
mlca_runtime.save_model(fitted, "model.pkl")

print(f"train rows={len(train)} features={len(feature_cols)} target={target}")
print(f"steps={list(fitted.named_steps)}")
print("saved model.pkl")
