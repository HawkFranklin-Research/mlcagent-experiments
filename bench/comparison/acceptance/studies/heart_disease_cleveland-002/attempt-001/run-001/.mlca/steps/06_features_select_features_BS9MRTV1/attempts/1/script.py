import mlca_runtime
import json
import pickle
import pandas as pd
from functools import partial
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

for key in ["targets", "excluded_columns", "task"]:
    if key not in protocol:
        raise KeyError(f"Missing key {key} in _protocol. Available keys: {list(protocol.keys())}")

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol.get("group_column", None)

with open(inputs["05_preprocess_fit_preprocessing_02TSADV2/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)

train = pd.read_parquet(inputs["train"])

if hasattr(preprocessor, "feature_names_in_"):
    feature_cols = list(preprocessor.feature_names_in_)
else:
    drop = set(targets) | set(excluded)
    if group_col:
        drop.add(group_col)
    candidate = [c for c in train.columns if c not in drop]
    feature_cols = []
    for c in candidate:
        s = train[c]
        if s.nunique(dropna=False) == len(s) and (
            s.dtype == object
            or pd.api.types.is_string_dtype(s)
            or (pd.api.types.is_integer_dtype(s) and not pd.api.types.is_float_dtype(s))
        ):
            continue
        feature_cols.append(c)

target_col = targets[0]
y = train[target_col]
X = train[feature_cols]

sample = X.head(1)
X_trans = preprocessor.transform(sample)
n_transformed = X_trans.shape[1]
k = min(20, n_transformed)

selector = SelectKBest(
    score_func=partial(mutual_info_classif, random_state=42),
    k=k,
)

if isinstance(preprocessor, Pipeline):
    steps = list(preprocessor.steps) + [("selector", selector)]
    pipe = Pipeline(steps)
else:
    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("selector", selector),
    ])

mlca_runtime.fit_record(pipe, X, partition="train", y=y)

mlca_runtime.save_model(pipe, "selector.pkl")

print(f"Raw features: {len(feature_cols)}")
print(f"Transformed features: {n_transformed}")
print(f"k: {k}")
print(f"Selected features: {pipe.named_steps['selector'].get_support().sum()}")
