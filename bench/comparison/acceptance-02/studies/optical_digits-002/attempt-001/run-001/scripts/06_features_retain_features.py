import json
import pickle
import pandas as pd
import mlca_runtime
from sklearn.compose import ColumnTransformer

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths["_protocol"].read_text())

for key in ("targets", "excluded_columns", "group_column"):
    if key not in protocol:
        raise KeyError(f"Missing protocol key {key!r}; available keys: {sorted(protocol)}")

targets = set(protocol["targets"] or [])
excluded = set(protocol["excluded_columns"] or [])
group = protocol["group_column"]
drop = targets | excluded
if group is not None:
    drop.add(group)

train = pd.read_parquet(paths["train"])

def key_like(s, n):
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return s.nunique(dropna=False) == n
    if pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return s.nunique(dropna=False) == n and (
            s.is_monotonic_increasing or s.is_monotonic_decreasing
        )
    return False

feature_cols = []
for col in train.columns:
    if col in drop:
        continue
    if key_like(train[col], len(train)):
        continue
    feature_cols.append(col)

X_raw = train.loc[:, feature_cols]

with open(paths["05_preprocess_fit_preprocessing_W49DB36F/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)

if hasattr(pre, "feature_names_in_"):
    expected = list(pre.feature_names_in_)
    missing = [c for c in expected if c not in X_raw.columns]
    if missing:
        raise KeyError(f"Preprocessor expects missing feature columns: {missing}")
    X_raw = X_raw.loc[:, expected]

out = pre.transform(X_raw)
if not isinstance(out, pd.DataFrame) and hasattr(out, "toarray"):
    out = out.toarray()

if isinstance(out, pd.DataFrame):
    pre_df = out.copy()
    pre_df.index = X_raw.index
else:
    names = None
    if hasattr(pre, "get_feature_names_out"):
        try:
            names = list(pre.get_feature_names_out())
        except Exception:
            names = None
    if names is None or len(names) != out.shape[1]:
        names = [f"f{i}" for i in range(out.shape[1])]
    pre_df = pd.DataFrame(out, index=X_raw.index, columns=names)

selector = ColumnTransformer(
    [("select_all", "passthrough", list(pre_df.columns))],
    remainder="drop",
)
if pre_df.shape[1] <= 300:
    selector.set_output(transform="pandas")

fitted = mlca_runtime.fit_record(selector, pre_df, partition="train")
mlca_runtime.save_model(fitted, "selector_all.pkl")
mlca_runtime.record_metric("n_features_selected", float(pre_df.shape[1]), "train")

print(f"train rows: {len(train)}")
print(f"raw feature columns: {len(feature_cols)}")
print(f"selector features: {pre_df.shape[1]}")
