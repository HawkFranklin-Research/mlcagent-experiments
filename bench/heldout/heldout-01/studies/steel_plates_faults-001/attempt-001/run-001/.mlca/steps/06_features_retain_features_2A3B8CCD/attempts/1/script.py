import json
import pickle

import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer

import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())


def req(d, key):
    if key not in d:
        raise KeyError(f"Missing protocol key {key!r}; available top-level keys: {sorted(d.keys())}")
    return d[key]


targets = list(req(protocol, "targets"))
excluded = list(req(protocol, "excluded_columns"))
group = req(protocol, "group_column")
_ = req(protocol, "task")

train = pd.read_parquet(inputs["train"])
drop = set(targets) | set(excluded)
if group is not None:
    drop.add(group)

feature_cols = []
for col in train.columns:
    if col in drop:
        continue
    s = train[col]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        if s.nunique(dropna=False) == len(s):
            continue
    elif pd.api.types.is_integer_dtype(s):
        if s.nunique(dropna=False) == len(s):
            name = col.lower()
            if any(tok in name for tok in ("id", "index", "key", "code")):
                continue
            vals = s.dropna().astype("int64")
            if len(vals) and vals.is_monotonic_increasing and vals.max() - vals.min() == len(vals) - 1:
                continue
    feature_cols.append(col)

pre = pickle.loads(inputs["05_preprocess_fit_preprocessing_CSS2QFXR/preprocessor.pkl"].read_bytes())
pre_cols = list(getattr(pre, "feature_names_in_", feature_cols))
missing = [c for c in pre_cols if c not in train.columns]
if missing:
    raise KeyError(f"Preprocessor expects missing columns: {missing}")

X_raw = train[pre_cols]
X_pre = pre.transform(X_raw)
if isinstance(X_pre, pd.DataFrame):
    pre_names = list(X_pre.columns)
else:
    try:
        pre_names = list(pre.get_feature_names_out())
    except Exception:
        pre_names = [f"f{i}" for i in range(X_pre.shape[1])]
    if sparse.issparse(X_pre):
        X_pre = X_pre.toarray()
    X_pre = pd.DataFrame(X_pre, columns=pre_names, index=train.index)

selector = ColumnTransformer(
    [("all", "passthrough", pre_names)],
    remainder="drop",
    verbose_feature_names_out=False,
)
if len(pre_names) <= 300:
    selector.set_output(transform="pandas")

mlca_runtime.fit_record(selector, X_pre, "train")
mlca_runtime.save_model(selector, "selector_all.pkl")
print(f"features={len(feature_cols)} preprocessor_inputs={len(pre_cols)} selected={len(pre_names)}")
