import json
import pickle
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif


def _get_key(d, key):
    if key not in d:
        raise KeyError(f"Missing key '{key}'. Available keys: {list(d.keys())}")
    return d[key]


mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs["_protocol"].read_text())
targets = _get_key(protocol, "targets")
excluded = _get_key(protocol, "excluded_columns")
group_col = _get_key(protocol, "group_column")
task = _get_key(protocol, "task")

train = pd.read_parquet(inputs["train"])
with open(inputs["05_preprocess_fit_preprocessing_XHDQ4AZ4/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)

drop_cols = set(targets) | set(excluded)
if group_col is not None:
    drop_cols.add(group_col)

candidate_cols = [c for c in train.columns if c not in drop_cols]
key_like = set()
for c in candidate_cols:
    s = train[c]
    if s.nunique(dropna=False) == len(train) and not pd.api.types.is_float_dtype(s):
        key_like.add(c)

feature_cols = [c for c in candidate_cols if c not in key_like]

expected = None
if hasattr(preprocessor, "feature_names_in_") and preprocessor.feature_names_in_ is not None:
    expected = list(preprocessor.feature_names_in_)
elif isinstance(preprocessor, Pipeline):
    first = list(preprocessor.named_steps.values())[0]
    if hasattr(first, "feature_names_in_") and first.feature_names_in_ is not None:
        expected = list(first.feature_names_in_)

if expected is not None:
    if set(expected) != set(feature_cols):
        raise ValueError(f"Preprocessor expects {expected}, but defined feature columns are {feature_cols}")
    feature_cols = expected

X = train[feature_cols]
y = train[targets[0]]

X_trans = preprocessor.transform(X)
n_features = X_trans.shape[1] if hasattr(X_trans, "shape") else len(X_trans[0])
k = min(20, n_features)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("selector", SelectKBest(mutual_info_classif, k=k)),
])

if n_features <= 300:
    pipeline.set_output(transform="pandas")

mlca_runtime.fit_record(pipeline, X, "train", y=y)
mlca_runtime.save_model(pipeline, "selector.pkl")

print(f"feature_cols: {len(feature_cols)}")
print(f"preprocessed features: {n_features}")
print(f"k: {k}")
print("saved selector.pkl")
