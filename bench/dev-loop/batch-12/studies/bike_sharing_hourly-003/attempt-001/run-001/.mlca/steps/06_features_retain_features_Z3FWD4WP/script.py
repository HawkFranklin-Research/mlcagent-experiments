import json
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())

targets = proto["targets"]
excluded = proto["excluded_columns"]
group = proto.get("group_column")

train = pd.read_parquet(inputs["train"])

def is_key_like(s: pd.Series) -> bool:
    if s.is_unique:
        if pd.api.types.is_string_dtype(s) or s.dtype == object:
            return True
        if pd.api.types.is_integer_dtype(s):
            return True
    return False

feature_cols = [
    c for c in train.columns
    if c not in targets and c not in excluded and c != group and not is_key_like(train[c])
]

selector = ColumnTransformer(
    [("passthrough", "passthrough", feature_cols)],
    remainder="drop",
    verbose_feature_names_out=False,
)
selector.set_output(transform="pandas")

X = train[feature_cols]
mlca_runtime.fit_record(selector, X, "train")
mlca_runtime.save_model(selector, "selector_all.pkl")

print(f"train_rows={len(train)}")
print(f"n_features={len(feature_cols)}")
print(f"selector_columns={list(selector.feature_names_in_)[:5]}...")
