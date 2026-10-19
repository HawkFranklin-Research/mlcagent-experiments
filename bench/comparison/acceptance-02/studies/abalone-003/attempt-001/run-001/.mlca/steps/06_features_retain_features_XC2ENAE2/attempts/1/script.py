import json, pickle
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

def need(key):
    if key not in protocol:
        raise KeyError(f"Missing '{key}'. Available top-level keys: {list(protocol.keys())}")
    return protocol[key]

targets = need("targets")
excluded = need("excluded_columns")
need("task")
group = protocol.get("group_column")

train = pd.read_parquet(inputs["train"])

pre_path = inputs["05_preprocess_fit_preprocessing_GTAJ2HZB/preprocessor.pkl"]
with open(pre_path, "rb") as f:
    pre = pickle.load(f)

if hasattr(pre, "feature_names_in_"):
    feature_cols = list(pre.feature_names_in_)
else:
    drop = set(targets) | set(excluded) | ({group} if isinstance(group, str) else set())
    n = len(train)
    feature_cols = []
    for c in train.columns:
        if c in drop:
            continue
        s = train[c]
        if (s.dtype == object or pd.api.types.is_string_dtype(s)) and s.nunique() == n:
            continue
        if pd.api.types.is_integer_dtype(s) and s.nunique() == n and s.min() == 0 and s.max() == n - 1:
            continue
        feature_cols.append(c)

forbidden = set(targets) | set(excluded) | ({group} if isinstance(group, str) else set())
feature_cols = [c for c in feature_cols if c not in forbidden]

missing = set(feature_cols) - set(train.columns)
if missing:
    raise KeyError(f"Feature columns missing from train: {sorted(missing)}")

ct = ColumnTransformer(
    [("keep", "passthrough", feature_cols)],
    remainder="drop",
    verbose_feature_names_out=False,
)
ct.set_output(transform="pandas")

mlca_runtime.fit_record(ct, train[feature_cols], "train", y=None)
mlca_runtime.save_model(ct, "selector_all.pkl")

print(f"n_features={len(feature_cols)} n_rows={len(train)}")
