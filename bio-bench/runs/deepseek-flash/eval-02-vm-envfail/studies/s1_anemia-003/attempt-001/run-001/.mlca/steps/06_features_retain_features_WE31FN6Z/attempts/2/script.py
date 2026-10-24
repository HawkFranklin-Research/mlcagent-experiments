import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
excluded = set(protocol["excluded_columns"])
group_col = protocol.get("group_column")
train = pd.read_parquet(inputs["train"])

all_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group_col]
feature_cols = []
for c in all_cols:
    s = train[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        if s.nunique(dropna=True) == len(train):
            continue
    if pd.api.types.is_integer_dtype(s):
        vals = s.dropna().sort_values().unique()
        if len(vals) == len(train) and len(vals) > 0 and np.array_equal(vals, np.arange(vals[0], vals[0] + len(vals))):
            continue
    feature_cols.append(c)

with open(inputs["05_preprocess_fit_preprocessing_T6A9FCB5/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)

if hasattr(pre, "feature_names_in_"):
    input_cols = [c for c in pre.feature_names_in_ if c not in targets and c not in excluded and c != group_col]
else:
    input_cols = feature_cols

X_train = train[input_cols]
selector = FunctionTransformer(validate=False)
final = Pipeline([("preprocessor", pre), ("selector", selector)])
final = mlca_runtime.fit_record(final, X_train, "train")
mlca_runtime.save_model(final, "selector_all.pkl")
print(f"Train shape: {train.shape}, feature cols: {len(input_cols)}")
print("Saved selector_all.pkl")
