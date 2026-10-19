import json
import pickle
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
for key in ["targets", "excluded_columns", "group_column", "task"]:
    if key not in protocol:
        raise ValueError(f"Missing {key}; available top-level keys: {list(protocol)}")

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group = protocol["group_column"]
train = pd.read_parquet(inputs["train"])

protected = set(targets) | set(excluded)
if group is not None:
    protected.add(group)

base_cols = [c for c in train.columns if c not in protected]
feature_cols = []
for c in base_cols:
    s = train[c]
    is_text = pd.api.types.is_string_dtype(s.dtype) or isinstance(s.dtype, pd.CategoricalDtype)
    if is_text and s.nunique(dropna=True) == len(s):
        continue
    feature_cols.append(c)

with open(inputs["06_preprocess_fit_preprocessing_09VH2GD4/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)

if hasattr(pre, "feature_names_in_"):
    expected = list(pre.feature_names_in_)
    bad = set(expected) & protected
    if bad:
        raise ValueError(f"Preprocessor input includes protected columns: {sorted(bad)}")
    missing = [c for c in expected if c not in train.columns]
    if missing:
        raise ValueError(f"Missing preprocessor input columns: {missing}")
    feature_cols = expected

X = train[feature_cols].copy()
y = train[targets[0]]

if hasattr(pre, "set_output"):
    try:
        pre.set_output(transform="pandas")
    except Exception:
        pass

n_features = len(feature_cols)
try:
    n_features = len(pre.get_feature_names_out())
except Exception:
    try:
        n_features = pre.transform(X.iloc[:1]).shape[1]
    except Exception:
        pass

selector = SelectKBest(score_func=mutual_info_classif, k=min(20, n_features))

if isinstance(pre, Pipeline):
    steps = [(name, est) for name, est in pre.steps if not isinstance(est, SelectKBest)]
    pipe = Pipeline(steps + [("select", selector)])
else:
    pipe = Pipeline([("preprocess", pre), ("select", selector)])

if hasattr(pipe, "set_output"):
    try:
        pipe.set_output(transform="pandas")
    except Exception:
        pass

fitted = mlca_runtime.fit_record(pipe, X, "train", y=y)
mlca_runtime.save_model(fitted, "selector.pkl")

sel = fitted.named_steps.get("select") if isinstance(fitted, Pipeline) else fitted
selected = int(sel.get_support().sum()) if hasattr(sel, "get_support") else min(20, n_features)
print(f"features={len(feature_cols)} k={min(20, n_features)} selected={selected}")
