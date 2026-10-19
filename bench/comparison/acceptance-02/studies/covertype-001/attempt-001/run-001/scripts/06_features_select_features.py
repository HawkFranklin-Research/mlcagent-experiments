import json
import pickle
import pandas as pd
import mlca_runtime
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())

needed = ("targets", "excluded_columns", "group_column", "task")
missing = [k for k in needed if k not in protocol]
if missing:
    raise KeyError(f"Missing _protocol keys {missing}; available top-level keys: {sorted(protocol)}")

targets = list(protocol["targets"])
excluded = set(protocol["excluded_columns"])
group = protocol["group_column"]
task = protocol["task"]

train = pd.read_parquet(paths["train"])
with open(paths["05_preprocess_fit_preprocessing_Z8NMCAXW/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)

candidates = [c for c in train.columns if c not in targets and c not in excluded and c != group]
feature_cols = []
for c in candidates:
    s = train[c]
    if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and s.nunique(dropna=False) == len(s):
        continue
    if pd.api.types.is_integer_dtype(s) and s.nunique(dropna=False) == len(s):
        name = str(c).lower()
        if "id" in name or (s.is_monotonic_increasing and s.diff().dropna().eq(1).all()):
            continue
    feature_cols.append(c)

if hasattr(pre, "feature_names_in_"):
    pre_cols = list(pre.feature_names_in_)
    bad = [c for c in pre_cols if c not in feature_cols]
    if bad:
        raise ValueError(f"Preprocessor expects columns outside the allowed feature set: {bad}")
    feature_cols = pre_cols

X = train[feature_cols].copy()
y = train[targets[0]].copy()

base = pre
if isinstance(base, Pipeline):
    steps = [(n, s) for n, s in base.steps if not isinstance(s, SelectKBest)]
    if len(steps) != len(base.steps):
        base = Pipeline(steps) if steps else "passthrough"

try:
    n_features = len(base.get_feature_names_out())
except Exception:
    if base == "passthrough":
        n_features = X.shape[1]
    else:
        mlca_runtime.fit_record(base, X, "train", y)
        n_features = len(base.get_feature_names_out())

k = min(20, n_features)
selector_pipeline = Pipeline([
    ("preprocess", base),
    ("selector", SelectKBest(score_func=mutual_info_classif, k=k)),
])

mlca_runtime.fit_record(selector_pipeline, X, "train", y)
mlca_runtime.save_model(selector_pipeline, "selector.pkl")

print("train", train.shape, "features", len(feature_cols))
print("preprocessed", n_features, "selected k", k)
print("saved selector.pkl")
