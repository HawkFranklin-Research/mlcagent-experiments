import json
import pandas as pd
import mlca_runtime
from sklearn.preprocessing import FunctionTransformer

mlca_runtime.set_seed(42)
ins = mlca_runtime.inputs()
protocol = json.loads(ins["_protocol"].read_text())

required = ["targets", "excluded_columns", "group_column", "task"]
missing = [k for k in required if k not in protocol]
if missing:
    raise KeyError(f"Missing protocol keys {missing}; available keys: {sorted(protocol.keys())}")

targets = list(protocol["targets"] or [])
excluded = list(protocol["excluded_columns"] or [])
group = protocol["group_column"]
exclude = set(targets) | set(excluded)
if group is not None:
    exclude.add(group)

train = pd.read_parquet(ins["train"])
exclude = exclude & set(train.columns)

def is_key_like(s: pd.Series) -> bool:
    if s.nunique(dropna=False) != len(s):
        return False
    if pd.api.types.is_integer_dtype(s.dtype):
        vals = set(s.dropna().unique())
        return not vals.issubset({0, 1})
    return (
        pd.api.types.is_object_dtype(s.dtype)
        or pd.api.types.is_string_dtype(s.dtype)
        or isinstance(s.dtype, pd.CategoricalDtype)
    )

feature_cols = [c for c in train.columns if c not in exclude and not is_key_like(train[c])]
X = train.loc[:, feature_cols]

selector = FunctionTransformer(feature_names_out="one-to-one")
fitted = mlca_runtime.fit_record(selector, X, partition="train")
mlca_runtime.save_model(fitted, "selector_all.pkl")
mlca_runtime.record_metric("n_features", float(len(feature_cols)), "train")
mlca_runtime.record_metric("n_train_rows", float(len(train)), "train")
print(f"rows={len(train)} features={len(feature_cols)}")
print("saved selector_all.pkl")
