import mlca_runtime, json
import pandas as pd
from sklearn.preprocessing import FunctionTransformer

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp["_protocol"].read_text())
required = ("targets", "excluded_columns", "task")
missing = [k for k in required if k not in proto]
if missing:
    raise KeyError(f"missing {missing}; available: {list(proto.keys())}")
targets = proto["targets"]
excluded = proto["excluded_columns"]
group_col = proto.get("group_column", None)

train = pd.read_parquet(inp["train"])
n = len(train)
cand = [c for c in train.columns if c not in targets and c not in excluded and c != group_col]

feat = []
for c in cand:
    s = train[c]
    if s.isna().all():
        feat.append(c)
        continue
    nn = int(s.dropna().shape[0])
    nun = int(s.nunique(dropna=True))
    if nn > 1 and nun == nn and pd.api.types.is_object_dtype(s):
        continue  # all-distinct text code
    if nn > 1 and nun == nn and pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s):
        continue  # all-distinct integer sequence
    feat.append(c)

selector = FunctionTransformer(validate=False)
mlca_runtime.fit_record(selector, train[feat], partition="train")
mlca_runtime.save_model(selector, "selector_all.pkl")
mlca_runtime.record_metric("n_features_selected", float(len(feat)), "train")
print(f"features={len(feat)} rows={n}")
