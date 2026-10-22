import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())


def req(d, key):
    if key not in d:
        raise KeyError(
            f"Missing protocol key {key!r}. Available top-level keys: {list(d.keys())}"
        )
    return d[key]


targets = req(protocol, "targets")
excluded_columns = req(protocol, "excluded_columns")
task = req(protocol, "task")
group_col = protocol.get("group_column")
target = targets[0]

train = pd.read_parquet(inputs["train"])

with open(inputs["05_preprocess_fit_preprocessing_5ED3RGCF/preprocessor.pkl"], "rb") as f:
    prep = pickle.load(f)

if hasattr(prep, "feature_names_in_"):
    feature_cols = list(prep.feature_names_in_)
else:
    def is_key_like(s):
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s.dtype):
            return bool(s.is_unique and s.nunique() == len(s))
        if pd.api.types.is_integer_dtype(s):
            if not s.is_unique or s.nunique() != len(s):
                return False
            name = s.name.lower()
            if any(tok in name for tok in ("id", "index", "instant", "key", "row")):
                return True
            vals = s.dropna().to_numpy()
            if len(vals) == len(s) and (vals.min() == 0 or vals.min() == 1):
                if np.array_equal(np.sort(vals), np.arange(vals.min(), vals.min() + len(vals))):
                    return True
        return False

    excluded = set(targets) | set(excluded_columns)
    if group_col is not None:
        excluded.add(group_col)
    feature_cols = [
        c for c in train.columns if c not in excluded and not is_key_like(train[c])
    ]

X_train = train[feature_cols]
y_train = train[target]

Xt = prep.transform(X_train)
if not isinstance(Xt, pd.DataFrame):
    try:
        names = prep.get_feature_names_out()
    except Exception:
        names = [f"f{i}" for i in range(Xt.shape[1])]
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    Xt = pd.DataFrame(Xt, index=X_train.index, columns=names)

selector = SelectKBest(score_func=f_regression, k="all")
if hasattr(selector, "set_output"):
    try:
        selector.set_output(transform="pandas")
    except Exception:
        pass

fitted = mlca_runtime.fit_record(selector, Xt, "train", y=y_train)
n_sel = int(fitted.get_support().sum())
mlca_runtime.record_metric("n_features_selected", float(n_sel), "train")
mlca_runtime.save_model(fitted, "selector_all.pkl")

print("train rows", len(train))
print("feature columns", len(feature_cols))
print("selected features", n_sel)
