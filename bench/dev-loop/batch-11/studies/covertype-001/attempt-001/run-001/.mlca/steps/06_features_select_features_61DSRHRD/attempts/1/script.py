import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

try:
    targets = list(protocol["targets"])
    excluded = list(protocol["excluded_columns"])
    _ = protocol["task"]
except KeyError as exc:
    raise KeyError(f"Missing protocol key {exc}; available top-level keys: {sorted(protocol)}") from None

group = protocol.get("group_column")
if len(targets) != 1:
    raise ValueError(f"Expected exactly one target, got {targets}")
target_col = targets[0]

train = pd.read_parquet(inputs["train"])
with open(inputs["05_preprocess_fit_preprocessing_XJYNH9KV/preprocessor.pkl"], "rb") as fh:
    pre = pickle.load(fh)

drop = set(targets) | set(excluded)
if group:
    drop.add(group)

feature_cols = []
for col in train.columns:
    if col in drop:
        continue
    s = train[col]
    is_text_like = s.dtype == object or pd.api.types.is_string_dtype(s)
    is_integer_like = pd.api.types.is_integer_dtype(s)
    if (is_text_like or is_integer_like) and s.nunique(dropna=False) == len(train):
        continue
    feature_cols.append(col)

if not feature_cols:
    raise ValueError("No feature columns remain after exclusions")

if hasattr(pre, "feature_names_in_"):
    raw_cols = list(pre.feature_names_in_)
    forbidden = set(targets) | set(excluded)
    if group:
        forbidden.add(group)
    if set(raw_cols) & forbidden:
        raise ValueError("Saved preprocessor expects a target or excluded column")
    missing_from_features = set(raw_cols) - set(feature_cols)
    if missing_from_features:
        raise ValueError(f"Saved preprocessor expects undeclared feature columns: {sorted(missing_from_features)}")
else:
    raw_cols = feature_cols

try:
    n_out = len(pre.get_feature_names_out())
except Exception:
    n_out = len(raw_cols)
if n_out < 1:
    raise ValueError("Preprocessing produced no output features")

k = min(20, n_out)
selector = SelectKBest(score_func=mutual_info_classif, k=k)
pipe = Pipeline([("preprocessor", pre), ("selectkbest", selector)])

X_train = train.loc[:, raw_cols]
y_train = train[target_col]
pipe = mlca_runtime.fit_record(pipe, X_train, "train", y=y_train)
mlca_runtime.save_model(pipe, "selector.pkl")

print(f"train rows={len(train)}, raw features={len(raw_cols)}, encoded features={n_out}, k={k}")
print("saved selector.pkl")
