import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import mlca_runtime

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()

def need(d, k):
    if k not in d:
        raise KeyError(f"missing {k!r}; available top-level keys: {sorted(d)}")
    return d[k]

protocol = json.loads(inp["_protocol"].read_text())
targets = need(protocol, "targets")
excluded = need(protocol, "excluded_columns")
group = need(protocol, "group_column")
if isinstance(targets, str):
    targets = [targets]
if isinstance(excluded, str):
    excluded = [excluded]

train = pd.read_parquet(inp["train"])

with open(inp["05_preprocess_fit_preprocessing_6KJVP4ST/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inp["06_features_select_features_TX9DR60B/selector.pkl"], "rb") as f:
    sel = pickle.load(f)

def input_names(obj):
    return list(obj.feature_names_in_) if hasattr(obj, "feature_names_in_") else None

pre_in = input_names(pre)
sel_in = input_names(sel)
sel_has_pre = (
    isinstance(sel, Pipeline)
    and sel_in is not None
    and pre_in is not None
    and sel_in == pre_in
)

clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
if sel_has_pre:
    pipe = Pipeline(list(sel.steps) + [("clf", clf)])
    raw_cols = sel_in
else:
    pipe = Pipeline([("pre", pre), ("sel", sel), ("clf", clf)])
    raw_cols = pre_in

if raw_cols is None:
    raise ValueError("could not determine raw input columns from saved preprocessing/selector")

forbidden = set(targets) | set(excluded) | ({group} if group else set())
bad = [c for c in raw_cols if c in forbidden]
if bad:
    raise ValueError(f"target/excluded/group columns would reach the model: {bad}")

X = train.loc[:, raw_cols]
y = train[targets[0]]
fitted = mlca_runtime.fit_record(pipe, X, partition="train", y=y)
mlca_runtime.save_model(fitted, "model.pkl")
print(f"features {len(raw_cols)}; rows {len(X)}")
print("saved model.pkl")
