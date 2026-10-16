import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

def get_protocol_key(protocol, key):
    if key not in protocol:
        raise KeyError(f"Missing key '{key}' in _protocol. Available keys: {list(protocol.keys())}")
    return protocol[key]

targets = get_protocol_key(protocol, "targets")
excluded = get_protocol_key(protocol, "excluded_columns")
group_col = get_protocol_key(protocol, "group_column")
task = get_protocol_key(protocol, "task")

train = pd.read_parquet(inputs["train"])

drop_cols = set(targets) | set(excluded)
if group_col is not None:
    drop_cols.add(group_col)

feature_cols = [c for c in train.columns if c not in drop_cols]

key_like = []
for c in feature_cols:
    s = train[c]
    if pd.api.types.is_string_dtype(s) or s.dtype == object:
        if s.nunique() == len(s):
            key_like.append(c)
    elif pd.api.types.is_integer_dtype(s):
        if s.nunique() == len(s):
            key_like.append(c)

feature_cols = [c for c in feature_cols if c not in key_like]

preprocessor_path = inputs["05_preprocess_fit_preprocessing_9D3FJJNM/preprocessor.pkl"]
with open(preprocessor_path, "rb") as f:
    preprocessor = pickle.load(f)

if hasattr(preprocessor, "feature_names_in_"):
    preproc_cols = list(preprocessor.feature_names_in_)
else:
    preproc_cols = feature_cols

missing = [c for c in preproc_cols if c not in train.columns]
if missing:
    raise ValueError(f"Preprocessor expects columns not in train: {missing}")

X_raw = train[preproc_cols].copy()

if hasattr(preprocessor, "set_output"):
    try:
        preprocessor.set_output(transform="pandas")
    except Exception:
        pass

X_trans = preprocessor.transform(X_raw)

if not isinstance(X_trans, pd.DataFrame):
    if hasattr(preprocessor, "get_feature_names_out"):
        names = preprocessor.get_feature_names_out()
        X_trans = pd.DataFrame(X_trans, columns=names, index=X_raw.index)
    else:
        raise ValueError("Preprocessor output is not a DataFrame and no get_feature_names_out available.")

selector = VarianceThreshold(threshold=0.0)
if hasattr(selector, "set_output"):
    selector.set_output(transform="pandas")

mlca_runtime.fit_record(selector, X_trans, partition="train")

mlca_runtime.save_model(selector, "selector_all.pkl")

print(f"Train shape: {train.shape}")
print(f"Feature columns: {len(feature_cols)}")
print(f"Preprocessor input columns: {len(preproc_cols)}")
print(f"Transformed shape: {X_trans.shape}")
print("Selector fitted and saved.")
