import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import mlca_runtime

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(Path(inp["_protocol"]).read_text())

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group = protocol.get("group_column")
if len(targets) != 1:
    raise ValueError("Expected exactly one target column")
target = targets[0]

train = pd.read_parquet(inp["train"])
drop = set(targets) | set(excluded) | ({group} if group else set())

def is_key_like(s: pd.Series) -> bool:
    name = str(s.name).lower()
    if (s.dtype == object or pd.api.types.is_string_dtype(s.dtype)) and s.notna().all():
        return s.nunique(dropna=True) == len(s)
    if pd.api.types.is_integer_dtype(s.dtype) and s.nunique(dropna=True) == len(s):
        return any(tok in name for tok in ("id", "code", "key"))
    return False

feature_cols = [
    c for c in train.columns
    if c not in drop and not is_key_like(train[c])
]

with open(inp["05_preprocess_fit_preprocessing_63H1R042/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inp["06_features_select_features_0WMEZ5WE/selector.pkl"], "rb") as f:
    sel = pickle.load(f)

def input_cols(obj):
    names = getattr(obj, "feature_names_in_", None)
    return list(names) if names is not None else None

def wraps_preprocessing(obj) -> bool:
    if not isinstance(obj, Pipeline):
        return False
    steps = list(obj.named_steps)
    return any(hasattr(obj.named_steps[s], "transform") for s in steps[:-1])

def checked_cols(cols):
    out = []
    for c in cols:
        if c not in train.columns or c in drop or c not in feature_cols:
            raise KeyError(f"Column not allowed for features: {c}")
        out.append(c)
    return out

sel_cols = input_cols(sel)
pre_cols = input_cols(pre)
clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)

if (
    sel is not None
    and isinstance(sel, Pipeline)
    and wraps_preprocessing(sel)
    and sel_cols is not None
    and set(sel_cols).issubset(set(feature_cols))
):
    X_cols = checked_cols(sel_cols)
    stages = [("selector", sel), ("clf", clf)]
else:
    stages = []
    if pre is not None:
        X_cols = checked_cols(pre_cols if pre_cols is not None else feature_cols)
        if hasattr(pre, "get_feature_names_out"):
            try:
                out_names = pre.get_feature_names_out()
                if len(out_names) <= 300 and hasattr(pre, "set_output"):
                    pre.set_output(transform="pandas")
            except Exception:
                pass
        stages.append(("preprocess", pre))
    elif sel is not None:
        X_cols = checked_cols(sel_cols if sel_cols is not None else feature_cols)
    else:
        raise RuntimeError("No preprocessing or selector artifact available")

    if sel is not None:
        stages.append(("select", sel))
    stages.append(("clf", clf))

X = train.loc[:, X_cols]
y = train[target]
pipe = Pipeline(stages)
model = mlca_runtime.fit_record(pipe, X, "train", y=y)
mlca_runtime.save_model(model, "model.pkl")

print("train", train.shape)
print("features", len(X_cols))
print("classes", len(getattr(model, "classes_", [])))
