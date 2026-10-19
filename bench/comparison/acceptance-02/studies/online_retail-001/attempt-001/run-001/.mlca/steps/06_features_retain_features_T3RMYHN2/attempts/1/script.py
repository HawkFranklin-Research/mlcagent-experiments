import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.preprocessing import FunctionTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

train = pd.read_parquet(inputs["train"])
_ = pd.read_parquet(inputs["val"])  # declared, not used for fitting

with open(inputs["05_preprocess_fit_preprocessing_HR6HVNXD/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)

feat_cols = None
if hasattr(preprocessor, "feature_names_in_"):
    feat_cols = list(preprocessor.feature_names_in_)
elif hasattr(preprocessor, "named_steps"):
    first = next(iter(preprocessor.named_steps.values()), None)
    if first is not None and hasattr(first, "feature_names_in_"):
        feat_cols = list(first.feature_names_in_)

if feat_cols is None:
    targets = list(protocol.get("targets", []))
    excluded = list(protocol.get("excluded_columns", []))
    group = protocol.get("group_column", None)
    drop = set(targets) | set(excluded)
    if group is not None:
        drop.add(group)
    feat_cols = [c for c in train.columns if c not in drop]

missing = [c for c in feat_cols if c not in train.columns]
if missing:
    raise KeyError(
        f"Missing feature columns {missing}. Available top-level protocol keys: "
        f"{sorted(protocol.keys())}"
    )

X_train = train[feat_cols]

selector = FunctionTransformer(func=None, validate=True, feature_names_out="one-to-one")
selector = mlca_runtime.fit_record(selector, X_train, partition="train")

mlca_runtime.save_model(selector, "selector_all.pkl")

out_names = list(selector.get_feature_names_out())
assert out_names == feat_cols, "feature names/order not preserved"
print(f"fit rows: {len(X_train)}, feature cols: {len(feat_cols)}")
print(f"first/last: {out_names[0]} / {out_names[-1]}")
