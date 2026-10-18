import mlca_runtime
import json
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())

def need(d, key):
    if key not in d:
        raise KeyError(f"Missing {key!r}; available top-level keys: {sorted(d)}")
    return d[key]

targets = list(need(protocol, "targets"))
excluded = list(need(protocol, "excluded_columns"))
task = need(protocol, "task")
_ = need(task, "primary_metric")
group_col = protocol.get("group_column")

train = pd.read_parquet(paths["train"])
missing_targets = [t for t in targets if t not in train.columns]
if missing_targets:
    raise KeyError(f"Missing target columns: {missing_targets}")

def is_key_like(s: pd.Series) -> bool:
    if s.nunique(dropna=False) != len(train):
        return False
    if pd.api.types.is_float_dtype(s):
        return False
    if pd.api.types.is_integer_dtype(s):
        return True
    if pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s):
        return True
    return isinstance(s.dtype, pd.CategoricalDtype)

target_set = set(targets)
excluded_set = set(excluded)
candidates = [
    c for c in train.columns
    if c not in target_set and c not in excluded_set and c != group_col
]
feature_cols = [c for c in candidates if not is_key_like(train[c])]
if not feature_cols:
    raise ValueError("No feature columns remain after exclusions and key-like detection")

num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c])]
cat_cols = [c for c in feature_cols if c not in num_cols]

transformers = []
if num_cols:
    transformers.append(("num", "passthrough", num_cols))
if cat_cols:
    transformers.append((
        "cat",
        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        cat_cols,
    ))

prep = ColumnTransformer(
    transformers,
    remainder="drop",
    verbose_feature_names_out=False,
).set_output(transform="pandas")

pipe = Pipeline([
    ("prep", prep),
    ("model", RandomForestRegressor(n_estimators=100, random_state=42)),
])

y_train = train[targets]
fitted = mlca_runtime.fit_record(pipe, train, partition="train", y=y_train)
mlca_runtime.save_model(fitted, "alternative_model.pkl")

print(f"train rows={len(train)}, feature columns={len(feature_cols)}")
print(f"targets={targets}")
print("saved alternative_model.pkl")
