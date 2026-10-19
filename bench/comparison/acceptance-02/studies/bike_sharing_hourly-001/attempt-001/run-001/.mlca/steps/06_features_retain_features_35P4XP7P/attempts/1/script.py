import mlca_runtime, json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
proto = json.loads(paths["_protocol"].read_text())

def need(k):
    if k not in proto:
        raise KeyError(f"Missing protocol key {k!r}; available top-level keys: {sorted(proto)}")
    return proto[k]

targets = list(need("targets"))
excluded = list(need("excluded_columns"))
group_col = need("group_column")
_ = need("task")

train = pd.read_parquet(paths["train"])
exclude = set(targets) | set(excluded)
if group_col is not None:
    exclude.add(group_col)

def key_like(s):
    if not (
        pd.api.types.is_object_dtype(s.dtype)
        or pd.api.types.is_string_dtype(s.dtype)
        or pd.api.types.is_integer_dtype(s.dtype)
    ):
        return False
    return s.notna().all() and s.nunique(dropna=False) == len(s)

base_cols = [c for c in train.columns if c not in exclude and not key_like(train[c])]

with open(paths["05_preprocess_fit_preprocessing_F9ABHBTW/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)

try:
    pre.set_output(transform="pandas")
except Exception:
    pass

feature_cols = list(getattr(pre, "feature_names_in_", base_cols))
for bad in exclude:
    if bad in feature_cols:
        raise ValueError(f"Excluded column {bad!r} found in preprocessor input schema")

X = train[feature_cols].copy()
X_trans = pre.transform(X)

if not isinstance(X_trans, pd.DataFrame):
    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()
    try:
        out_cols = list(pre.get_feature_names_out(feature_cols))
    except Exception:
        out_cols = [f"f{i}" for i in range(X_trans.shape[1])]
    X_trans = pd.DataFrame(X_trans, index=X.index, columns=out_cols)

selector = FunctionTransformer(validate=False, feature_names_out="one-to-one")
selector = mlca_runtime.fit_record(selector, X_trans, partition="train")
full = Pipeline([("preprocessor", pre), ("selector", selector)])
mlca_runtime.save_model(full, "selector_all.pkl")
print("feature_cols", len(feature_cols))
print("train_rows", len(train))
print("selector_saved")
