import json
import pickle
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
train = pd.read_parquet(inputs["train"])
with open(inputs["05_preprocess_fit_preprocessing_SMPGAWB4/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)

for key in ("targets", "excluded_columns", "group_column"):
    if key not in protocol:
        raise KeyError(f"{key!r} missing; available top-level keys: {sorted(protocol)}")

targets = list(protocol["targets"])
excluded = list(protocol["excluded_columns"])
group = protocol["group_column"]
drop = set(targets) | set(excluded)
if group is not None:
    drop |= set(group) if isinstance(group, (list, tuple, set)) else {group}

n = len(train)
rule_cols = []
for c in train.columns:
    if c in drop:
        continue
    s = train[c]
    if s.nunique(dropna=False) == n and (
        pd.api.types.is_object_dtype(s)
        or pd.api.types.is_string_dtype(s)
        or isinstance(s.dtype, pd.CategoricalDtype)
        or pd.api.types.is_integer_dtype(s)
    ):
        continue
    rule_cols.append(c)

pcols_set = None
if hasattr(preprocessor, "feature_names_in_"):
    cand = set(preprocessor.feature_names_in_)
    if cand and cand.issubset(set(train.columns)):
        pcols_set = cand

if pcols_set is not None:
    feature_cols = [c for c in rule_cols if c in pcols_set]
    if not feature_cols:
        feature_cols = rule_cols
else:
    feature_cols = rule_cols

if not feature_cols:
    raise ValueError("No feature columns remain after exclusions and key-like filtering.")

X_raw = train[feature_cols].copy()
selector = FunctionTransformer(validate=False)
fitted = mlca_runtime.fit_record(selector, X_raw, "train")
mlca_runtime.save_model(fitted, "selector_all.pkl")
mlca_runtime.record_fact_candidate("selector_features", int(X_raw.shape[1]), source="selector_all.pkl")
print(f"rows: {len(X_raw)} features: {X_raw.shape[1]}")
