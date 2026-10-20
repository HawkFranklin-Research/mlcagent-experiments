import mlca_runtime, json
import pandas as pd
from sklearn.preprocessing import FunctionTransformer

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())
train = pd.read_parquet(inp["train"])

for k in ("targets", "excluded_columns", "task"):
    if k not in proto:
        raise KeyError(f"missing {k}; available top-level keys: {list(proto.keys())}")
targets = list(proto["targets"])
excluded = list(proto["excluded_columns"])
group = proto.get("group_column", None)

def is_keylike(s):
    if s.dtype.kind in "iu" or s.dtype == object or str(s.dtype).startswith("string") or str(s.dtype) == "category":
        nonnull = s.dropna()
        if len(nonnull) == 0:
            return False
        return nonnull.is_unique
    return False

skip = set(targets) | set(excluded)
if group:
    skip.add(group)
features = [c for c in train.columns if c not in skip and not is_keylike(train[c])]

X = train[features]

selector = FunctionTransformer(validate=False)
fitted = mlca_runtime.fit_record(selector, X, "train")

mlca_runtime.save_model(fitted, "selector_all.pkl")
print(f"features={len(features)} rows={len(X)}")
