import mlca_runtime, json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())

try:
    targets = list(proto["targets"])
    excluded = list(proto.get("excluded_columns", []))
    group_col = proto.get("group_column")
except KeyError as e:
    raise KeyError(f"Missing protocol key: {e}. Available top-level keys: {list(proto.keys())}")

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)
feat_cols = [c for c in train.columns if c not in drop]
key_like = []
for c in feat_cols:
    s = train[c]
    if s.dtype.kind in "OU" and s.nunique() == len(s):
        key_like.append(c)
    elif pd.api.types.is_integer_dtype(s) and s.nunique() == len(s):
        key_like.append(c)
feat_cols = [c for c in feat_cols if c not in key_like]

candidates = []
seen = set()
for label, key in [("logistic", "07_model_train_logistic_CM1VEX5J/model.pkl"),
                   ("alternative", "08_model_train_alternative_3YZTFRCC/alternative_model.pkl"),
                   ("latest", "latest_model")]:
    p = inputs[key]
    rp = p.resolve()
    if rp in seen:
        continue
    seen.add(rp)
    with open(p, "rb") as f:
        candidates.append((label, pickle.load(f)))

def expected_columns(model):
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if isinstance(model, Pipeline):
        for step in model.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return None

out = pd.DataFrame({"row_id": val.index, "partition": "validation"}, index=val.index)
for t in targets:
    out[t] = val[t]

for label, model in candidates:
    cols = expected_columns(model) or feat_cols
    missing = [c for c in cols if c not in val.columns]
    if missing:
        raise KeyError(f"Validation missing columns for {label}: {missing}")
    out[f"pred_{label}"] = model.predict(val[cols])

mlca_runtime.save_row_outputs(out, "validation_predictions.csv")
print(f"rows={len(out)} candidates={len(candidates)} cols={list(out.columns)}")
