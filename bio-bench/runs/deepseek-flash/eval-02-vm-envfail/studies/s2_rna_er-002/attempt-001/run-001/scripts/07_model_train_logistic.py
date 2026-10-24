import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
train = pd.read_parquet(inputs["train"])
protocol = json.loads(inputs["_protocol"].read_text())

targets = list(protocol["targets"])
excluded = set(protocol.get("excluded_columns", []))
group = protocol.get("group_column")
if len(targets) != 1:
    raise ValueError(f"expected one target, got {targets}")
target = targets[0]
if target not in train.columns:
    raise ValueError(f"target {target!r} not in train columns")
y = train[target]

prep = pickle.loads(inputs["05_preprocess_fit_preprocessing_03EVNKD3/preprocessor.pkl"].read_bytes())
sel = pickle.loads(inputs["06_features_select_features_9KY0CV9N/selector.pkl"].read_bytes())

def feat_cols(obj):
    return list(obj.feature_names_in_) if hasattr(obj, "feature_names_in_") else None

raw_cols = feat_cols(prep)
if raw_cols is None and isinstance(sel, Pipeline) and len(sel.named_steps) > 1:
    raw_cols = feat_cols(sel)

if raw_cols is None:
    drop = set(targets) | excluded | ({group} if group else set())
    raw_cols = []
    for c in train.columns:
        if c in drop:
            continue
        s = train[c]
        nun = s.nunique(dropna=False)
        if nun == len(train) and nun > 1 and (
            pd.api.types.is_object_dtype(s)
            or pd.api.types.is_integer_dtype(s)
            or pd.api.types.is_string_dtype(s)
        ):
            continue
        raw_cols.append(c)

forbidden = set(targets) | excluded | ({group} if group else set())
bad = set(raw_cols) & forbidden
if bad:
    raise ValueError(f"feature columns include forbidden columns: {sorted(bad)}")
missing = [c for c in raw_cols if c not in train.columns]
if missing:
    raise ValueError(f"feature columns missing from train: {missing}")

sel_frontend = (
    isinstance(sel, Pipeline)
    and len(sel.named_steps) > 1
    and hasattr(sel, "feature_names_in_")
    and set(sel.feature_names_in_) == set(raw_cols)
)

clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
if sel_frontend:
    model = Pipeline([("features", sel), ("clf", clf)])
else:
    steps = []
    if prep is not None:
        steps.append(("preprocessor", prep))
    if sel is not None:
        steps.append(("selector", sel))
    steps.append(("clf", clf))
    model = Pipeline(steps)

X = train[raw_cols]
model = mlca_runtime.fit_record(model, X, "train", y=y)
mlca_runtime.save_model(model, "model.pkl")
print(f"train={train.shape} features={len(raw_cols)}")
