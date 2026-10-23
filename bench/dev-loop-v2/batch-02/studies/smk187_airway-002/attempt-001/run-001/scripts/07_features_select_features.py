import json
import pickle

import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline

import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs["_protocol"].read_text())


def require(d, key):
    if key not in d:
        raise KeyError(
            f"Missing protocol key {key!r}; available top-level keys: {sorted(d)}"
        )
    return d[key]


targets = require(protocol, "targets")
excluded = require(protocol, "excluded_columns")
task = require(protocol, "task")
group_col = protocol.get("group_column")
if not targets:
    raise ValueError("protocol['targets'] is empty")
target = targets[0]

train = pd.read_parquet(inputs["train"])
if target not in train.columns:
    raise KeyError(f"target {target!r} not found in train columns")

forbidden = set(targets) | set(excluded)
if group_col:
    forbidden.add(group_col)

feature_cols = [c for c in train.columns if c not in forbidden]

drop_key = []
n_rows = len(train)
for c in list(feature_cols):
    s = train[c]
    if n_rows == 0 or s.nunique(dropna=True) != n_rows:
        continue
    name = str(c).lower()
    if (
        pd.api.types.is_object_dtype(s)
        or pd.api.types.is_string_dtype(s)
        or isinstance(s.dtype, pd.CategoricalDtype)
    ):
        drop_key.append(c)
    elif pd.api.types.is_integer_dtype(s) and any(
        tok in name for tok in ("id", "key", "code", "index")
    ):
        drop_key.append(c)
feature_cols = [c for c in feature_cols if c not in drop_key]

with open(inputs["06_preprocess_fit_preprocessing_XWW5WTFN/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)

saved_names = None
if hasattr(preprocessor, "feature_names_in_"):
    saved_names = list(preprocessor.feature_names_in_)
elif isinstance(preprocessor, Pipeline) and preprocessor.steps:
    first = preprocessor.steps[0][1]
    if hasattr(first, "feature_names_in_"):
        saved_names = list(first.feature_names_in_)

if saved_names is not None:
    bad = set(saved_names) & forbidden
    if bad:
        raise ValueError(f"saved preprocessor expects forbidden columns: {sorted(bad)}")
    missing = [c for c in saved_names if c not in train.columns]
    if missing:
        raise KeyError(f"saved preprocessor expects missing columns: {missing}")
    extra = [c for c in saved_names if c not in feature_cols]
    if extra:
        raise ValueError(f"saved preprocessor expects non-feature columns: {extra}")
    X_cols = saved_names
else:
    X_cols = feature_cols

X_train = train.loc[:, X_cols]
y_train = train[target]

n_transformed = None
try:
    n_transformed = len(preprocessor.get_feature_names_out())
except Exception:
    try:
        n_transformed = preprocessor.transform(X_train.head(1)).shape[1]
    except Exception:
        n_transformed = len(feature_cols)

if not n_transformed or n_transformed < 1:
    raise ValueError("could not determine transformed feature count")

if n_transformed <= 300 and hasattr(preprocessor, "set_output"):
    try:
        preprocessor.set_output(transform="pandas")
    except Exception:
        pass

k = max(1, min(20, n_transformed))
selector = SelectKBest(score_func=mutual_info_classif, k=k)

if isinstance(preprocessor, Pipeline):
    pipe = Pipeline(list(preprocessor.steps) + [("select_k_best_mi", selector)])
else:
    pipe = Pipeline([("preprocessor", preprocessor), ("select_k_best_mi", selector)])

mlca_runtime.fit_record(pipe, X_train, "train", y=y_train)
mlca_runtime.save_model(pipe, "selector.pkl")

print(f"train rows={len(train)}, candidate features={len(feature_cols)}, used={len(X_cols)}")
print(f"transformed features={n_transformed}, k={k}")
print("saved selector.pkl")
