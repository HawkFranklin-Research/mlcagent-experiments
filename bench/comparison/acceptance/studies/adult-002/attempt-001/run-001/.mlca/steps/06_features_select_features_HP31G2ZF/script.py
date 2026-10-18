import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

for key in ("targets", "excluded_columns", "task"):
    if key not in protocol:
        raise KeyError(f"Missing protocol key {key!r}; available keys: {sorted(protocol)}")

target_cols = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group_col = protocol.get("group_column")
protected = set(target_cols) | set(excluded)
if group_col:
    protected.add(group_col)

train = pd.read_parquet(inputs["train"])

def is_key_like(s: pd.Series) -> bool:
    n = len(s)
    if n == 0:
        return False
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return s.nunique(dropna=False) == n
    if pd.api.types.is_integer_dtype(s):
        return s.nunique(dropna=False) == n and s.is_monotonic_increasing
    return False

feature_cols = [c for c in train.columns if c not in protected and not is_key_like(train[c])]

with open(inputs["05_preprocess_fit_preprocessing_5C9G845H/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)

def configure(obj) -> None:
    if isinstance(obj, Pipeline):
        try:
            obj.set_output(transform="pandas")
        except Exception:
            pass
        for _, step in obj.steps:
            configure(step)
    elif isinstance(obj, ColumnTransformer):
        try:
            obj.set_output(transform="pandas")
        except Exception:
            pass
        for _, trans, _ in obj.transformers:
            if trans not in ("passthrough", "drop"):
                configure(trans)
    elif isinstance(obj, OneHotEncoder):
        obj.set_params(handle_unknown="ignore", sparse_output=False)

configure(pre)

expected = getattr(pre, "feature_names_in_", None)
if expected is not None:
    expected = list(expected)
    bad = set(expected) & protected
    if bad:
        raise ValueError(f"Preprocessor expects protected columns: {sorted(bad)}")
    if set(expected) != set(feature_cols):
        raise ValueError(
            f"Preprocessor feature mismatch: expected {len(expected)} cols, computed {len(feature_cols)}"
        )
    feature_cols = expected

X_train = train.loc[:, feature_cols].copy()
y_train, _ = pd.factorize(train[target_cols[0]])

selector = SelectKBest(score_func=mutual_info_classif, k=20)

if isinstance(pre, Pipeline):
    base_steps = [(name, step) for name, step in pre.steps if not isinstance(step, SelectKBest)]
    sel_name = "selector"
    names = {name for name, _ in base_steps}
    while sel_name in names:
        sel_name += "_"
    pipe = Pipeline(base_steps + [(sel_name, selector)])
else:
    sel_name = "selector"
    pipe = Pipeline([("preprocess", pre), (sel_name, selector)])

try:
    pipe.set_output(transform="pandas")
except Exception:
    pass

fitted = mlca_runtime.fit_record(pipe, X_train, "train", y=y_train)
mlca_runtime.save_model(fitted, "selector.pkl")

selected = int(fitted.named_steps[sel_name].get_support().sum())
print(f"features: {len(feature_cols)}")
print(f"selected: {selected}")
print("saved selector.pkl")
