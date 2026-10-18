import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
train = pd.read_parquet(inp["train"])

protocol = json.loads(inp["_protocol"].read_text())
try:
    targets = protocol["targets"]
    excluded = protocol["excluded_columns"]
    task = protocol["task"]
except KeyError as e:
    raise KeyError(f"Missing protocol key {e}; available keys: {list(protocol)}") from e
group = protocol.get("group_column")
if len(targets) != 1:
    raise ValueError(f"Expected one target, got {targets}")
target = targets[0]

with open(inp["05_preprocess_fit_preprocessing_VDSDFD6K/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inp["06_features_select_features_C9CJJ4J0/selector.pkl"], "rb") as f:
    sel = pickle.load(f)

def input_cols(obj):
    if obj is None:
        return None
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        for step in obj.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return None

bad = set(targets) | set(excluded)
if group:
    bad.add(group)
derived = [c for c in train.columns if c not in bad]
key_like = []
for c in derived:
    s = train[c]
    if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or pd.api.types.is_integer_dtype(s)) and s.nunique(dropna=True) == len(train):
        key_like.append(c)
derived = [c for c in derived if c not in key_like]

pre_cols = input_cols(pre)
sel_cols = input_cols(sel)
raw_cols = pre_cols or sel_cols or derived
raw_cols = [c for c in raw_cols if c in train.columns and c not in bad]
if not raw_cols:
    raise ValueError("No feature columns remain after exclusions")

if isinstance(sel, Pipeline) and sel_cols is not None and set(sel_cols) == set(raw_cols):
    base = sel
else:
    base = Pipeline([("preprocessor", pre), ("selector", sel)])

model_name = "logreg"
if model_name in base.named_steps:
    model_name = "classifier"
full = Pipeline(list(base.steps) + [
    (model_name, LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))
])

X = train.loc[:, raw_cols]
y = train[target]
fitted = mlca_runtime.fit_record(full, X, "train", y=y)
mlca_runtime.save_model(fitted if fitted is not None else full, "model.pkl")
print("train", train.shape, "features", len(raw_cols))
print("steps", list(full.named_steps))
