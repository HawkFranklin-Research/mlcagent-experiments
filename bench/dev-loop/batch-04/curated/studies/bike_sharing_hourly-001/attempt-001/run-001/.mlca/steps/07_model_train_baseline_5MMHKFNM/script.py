import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())

for key in ["targets", "excluded_columns", "task"]:
    if key not in proto:
        raise KeyError(f"Missing '{key}' in _protocol. Available: {list(proto.keys())}")

targets = list(proto["targets"])
excluded = list(proto["excluded_columns"])
task = proto["task"]
group = proto.get("group_column")
primary_metric = task.get("primary_metric")

train_df = pd.read_parquet(inputs["train"])

base_cols = [c for c in train_df.columns if c not in targets and c not in excluded and c != group]
key_like = []
for c in base_cols:
    s = train_df[c]
    if (pd.api.types.is_string_dtype(s) or s.dtype == object) and s.nunique(dropna=False) == len(s) and len(s) > 1:
        key_like.append(c)
    elif pd.api.types.is_integer_dtype(s) and s.nunique(dropna=False) == len(s) and len(s) > 1:
        key_like.append(c)
feature_cols = [c for c in base_cols if c not in key_like]

with open(inputs["05_preprocess_fit_preprocessing_DVNTE848/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inputs["06_features_retain_features_TBVVBD2H/selector_all.pkl"], "rb") as f:
    sel = pickle.load(f)

def maybe_pandas(obj):
    if hasattr(obj, "get_feature_names_out") and hasattr(obj, "set_output"):
        try:
            if len(obj.get_feature_names_out()) <= 300:
                obj.set_output(transform="pandas")
        except Exception:
            pass

maybe_pandas(pre)
maybe_pandas(sel)

def contains_preprocessing(obj):
    if isinstance(obj, Pipeline):
        for name, step in obj.named_steps.items():
            if isinstance(step, ColumnTransformer):
                return True
            if "pre" in name.lower() or "encode" in name.lower() or "scale" in name.lower():
                return True
    return False

if contains_preprocessing(sel):
    steps = [("frontend", sel), ("model", LinearRegression())]
    frontend = sel
else:
    steps = [("preprocessor", pre), ("selector", sel), ("model", LinearRegression())]
    frontend = pre

if hasattr(frontend, "feature_names_in_"):
    input_cols = list(frontend.feature_names_in_)
else:
    input_cols = feature_cols

forbidden = set(targets) | set(excluded) | ({group} if group else set()) | set(key_like)
bad = set(input_cols) & forbidden
if bad:
    raise ValueError(f"Frontend expects forbidden columns: {bad}")
input_cols = [c for c in input_cols if c in train_df.columns]

X = train_df[input_cols].copy()
y = train_df[targets].copy()
pipe = Pipeline(steps)
fitted = mlca_runtime.fit_record(pipe, X, "train", y=y)
mlca_runtime.save_model(fitted, "model.pkl")

print(f"Train rows: {len(train_df)}")
print(f"Features: {len(input_cols)}")
print(f"Targets: {len(targets)}")
print("Saved model.pkl")
