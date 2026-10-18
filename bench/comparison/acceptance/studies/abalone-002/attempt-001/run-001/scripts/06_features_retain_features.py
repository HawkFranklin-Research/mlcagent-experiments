import json
import pickle
import mlca_runtime
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, f_regression

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()

def need(d, key):
    if key not in d:
        raise KeyError(f"Missing {key!r}; available keys: {sorted(d)}")
    return d[key]

protocol = json.loads(paths["_protocol"].read_text())
task = need(protocol, "task")
targets = list(need(protocol, "targets"))
excluded = list(need(protocol, "excluded_columns"))
group_col = need(protocol, "group_column")

train = pd.read_parquet(paths["train"])
manifest = json.loads(paths["05_preprocess_fit_preprocessing_FG7YG4JM/manifest.json"].read_text())
with open(paths["05_preprocess_fit_preprocessing_FG7YG4JM/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)

protected = set(targets) | set(excluded)
if group_col is not None:
    protected.add(group_col)

feature_cols = []
for col in train.columns:
    if col in protected:
        continue
    s = train[col]
    unique = len(train) > 0 and s.nunique(dropna=False) == len(train)
    if unique and (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
        continue
    if unique and pd.api.types.is_integer_dtype(s):
        name = str(col).lower()
        if s.is_monotonic_increasing or any(t in name for t in ("id", "key", "index")):
            continue
    feature_cols.append(col)

expected = getattr(pre, "feature_names_in_", None)
if expected is None and isinstance(pre, Pipeline) and pre.steps:
    expected = getattr(pre.steps[0][1], "feature_names_in_", None)
if expected is not None:
    expected = [str(c) for c in expected]
    if set(expected) != set(feature_cols):
        raise ValueError("Saved preprocessor input columns do not match computed feature columns")
    feature_cols = expected

bad = set(feature_cols) & protected
if bad:
    raise ValueError(f"Protected columns reached feature list: {sorted(bad)}")

X = train.loc[:, feature_cols]
y = train[targets[0]] if len(targets) == 1 else train[targets]

score_func = f_regression if str(task.get("type", "")).lower() == "regression" else f_classif
selector = SelectKBest(score_func=score_func, k="all")
try:
    selector.set_output(transform="pandas")
except Exception:
    pass

pipeline = Pipeline([
    ("preprocessor", pre),
    ("selector", selector),
])

fitted = mlca_runtime.fit_record(pipeline, X, "train", y=y)
mlca_runtime.save_model(fitted, "selector_all.pkl")

print(f"train rows: {len(train)}")
print(f"features: {len(feature_cols)}")
print("saved selector_all.pkl")
