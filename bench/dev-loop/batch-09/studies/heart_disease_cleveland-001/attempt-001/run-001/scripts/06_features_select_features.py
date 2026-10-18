import json
import pickle
import functools
import mlca_runtime
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

for key in ("targets", "excluded_columns", "group_column"):
    if key not in protocol:
        raise KeyError(f"Missing {key!r}; available top-level keys: {sorted(protocol)}")

targets = protocol["targets"]
excluded_columns = protocol["excluded_columns"]
group_column = protocol["group_column"]

train = pd.read_parquet(inputs["train"])
with open(inputs["05_preprocess_fit_preprocessing_G1EGCAW6/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)

if len(targets) != 1:
    raise ValueError(f"Expected one target, got {targets}")
target = targets[0]
if target not in train.columns:
    raise ValueError(f"Target {target!r} not in training table")

exclude = set(targets) | set(excluded_columns)
if group_column is not None:
    exclude.add(group_column)

def is_key_like(series, name):
    n = len(series)
    if n == 0 or series.isna().any() or series.nunique(dropna=False) != n:
        return False
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        return True
    if pd.api.types.is_integer_dtype(series):
        lname = name.lower()
        return series.is_monotonic_increasing or series.is_monotonic_decreasing or "id" in lname or "key" in lname
    return False

candidate_cols = [c for c in train.columns if c not in exclude]
feature_cols = [c for c in candidate_cols if not is_key_like(train[c], c)]

if hasattr(preprocessor, "feature_names_in_"):
    prep_cols = list(preprocessor.feature_names_in_)
    bad = set(prep_cols) & exclude
    if bad:
        raise ValueError(f"Preprocessor expects excluded/target columns: {sorted(bad)}")
    if set(prep_cols) != set(feature_cols):
        missing = sorted(set(prep_cols) - set(feature_cols))
        extra = sorted(set(feature_cols) - set(prep_cols))
        raise ValueError(f"Preprocessor feature mismatch missing={missing} extra={extra}")
    feature_cols = prep_cols

X = train.loc[:, feature_cols].copy()
y = train[target]

try:
    n_features = len(preprocessor.get_feature_names_out())
except Exception:
    n_features = preprocessor.transform(X).shape[1]

if n_features <= 300 and hasattr(preprocessor, "set_output"):
    try:
        preprocessor.set_output(transform="pandas")
    except Exception:
        pass

k = min(20, int(n_features))
if k < 1:
    raise ValueError("No transformed features available for SelectKBest")
selector = SelectKBest(
    score_func=functools.partial(mutual_info_classif, random_state=42),
    k=k,
)

select_name = "select_k_best"
if isinstance(preprocessor, Pipeline):
    existing_names = [name for name, _ in preprocessor.steps]
    while select_name in existing_names:
        select_name += "_new"
    final = Pipeline(steps=list(preprocessor.steps) + [(select_name, selector)])
else:
    final = Pipeline(steps=[("preprocess", preprocessor), (select_name, selector)])

if n_features <= 300 and hasattr(final, "set_output"):
    try:
        final.set_output(transform="pandas")
    except Exception:
        pass

fitted = mlca_runtime.fit_record(final, X, "train", y=y)
n_selected = int(fitted.named_steps[select_name].get_support().sum())

out_path = mlca_runtime.save_model(fitted, "selector.pkl")
mlca_runtime.record_metric("n_features_input", len(feature_cols), "train")
mlca_runtime.record_metric("n_features_transformed", int(n_features), "train")
mlca_runtime.record_metric("n_selected_features", n_selected, "train")
print(f"train rows={len(train)} feature columns={len(feature_cols)} transformed width={n_features} k={k}")
print(f"saved {out_path.name} selected={n_selected}")
