import mlca_runtime
import json
import pickle
from functools import partial
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

for key in ("targets", "excluded_columns", "group_column", "task"):
    if key not in protocol:
        raise KeyError(f"Missing protocol key {key!r}; available top-level keys: {sorted(protocol)}")

targets = protocol["targets"]
excluded = set(protocol["excluded_columns"] or [])
group = protocol["group_column"]
group_col = group if group not in (None, "") else None
if len(targets) != 1:
    raise ValueError(f"Expected exactly one target, got {targets!r}")
target = targets[0]

train = pd.read_parquet(inputs["train"])

def is_key_like(s: pd.Series) -> bool:
    n = len(s)
    if n == 0 or s.nunique(dropna=True) != n:
        return False
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return True
    if pd.api.types.is_integer_dtype(s):
        return bool(s.is_monotonic_increasing or s.is_monotonic_decreasing)
    return False

target_set = set(targets)
base_features = [
    c for c in train.columns
    if c not in target_set and c not in excluded and c != group_col and not is_key_like(train[c])
]

with open(inputs["05_preprocess_fit_preprocessing_A5VJ9RPH/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)

if hasattr(pre, "feature_names_in_"):
    pre_cols = list(pre.feature_names_in_)
    missing = [c for c in pre_cols if c not in train.columns]
    if missing:
        raise ValueError(f"Preprocessor expects missing columns: {missing}")
    bad = [c for c in pre_cols if c in target_set or c in excluded or c == group_col]
    if bad:
        raise ValueError(f"Preprocessor expects outcome/excluded/group columns: {bad}")
    extra = [c for c in pre_cols if c not in base_features]
    if extra:
        raise ValueError(f"Preprocessor expects non-feature/key columns: {extra}")
    feature_cols = pre_cols
else:
    feature_cols = base_features

X = train.loc[:, feature_cols].copy()
y = train[target].copy()

def output_width(preprocessor, X_df):
    try:
        return len(preprocessor.get_feature_names_out())
    except Exception:
        return preprocessor.transform(X_df.head(1)).shape[1]

n_features = output_width(pre, X)
if n_features < 1:
    raise ValueError("Preprocessing produced no features")

if n_features <= 300:
    try:
        pre.set_output(transform="pandas")
    except Exception:
        pass

k = min(20, n_features)
selector = SelectKBest(score_func=partial(mutual_info_classif, random_state=42), k=k)

if isinstance(pre, Pipeline):
    if any(isinstance(step, SelectKBest) for _, step in pre.steps):
        raise ValueError("Loaded preprocessor already contains a SelectKBest step")
    pipe = Pipeline(steps=list(pre.steps) + [("selectkbest", selector)])
else:
    pipe = Pipeline([("preprocess", pre), ("selectkbest", selector)])

if n_features <= 300:
    try:
        pipe.set_output(transform="pandas")
    except Exception:
        pass

fitted = mlca_runtime.fit_record(pipe, X, "train", y)
mlca_runtime.save_model(fitted, "selector.pkl")
mlca_runtime.record_metric("n_features_preprocessed", int(n_features), "train")
mlca_runtime.record_metric("n_features_selected", int(k), "train")
print(f"train={train.shape} features={len(feature_cols)}")
print(f"preprocessed={n_features} selected={k}")
