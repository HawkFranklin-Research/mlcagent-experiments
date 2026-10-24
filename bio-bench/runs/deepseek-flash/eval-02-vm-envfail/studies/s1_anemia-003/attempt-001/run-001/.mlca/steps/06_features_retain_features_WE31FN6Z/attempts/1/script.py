import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
out_dir = mlca_runtime.output_dir()

protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol.get("group_column")
task = protocol["task"]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

all_cols = [c for c in train.columns if c not in targets and c not in excluded and (group_col is None or c != group_col)]
feature_cols = []
for c in all_cols:
    s = train[c]
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        if s.nunique(dropna=True) == len(train):
            continue
    feature_cols.append(c)

with open(inputs["05_preprocess_fit_preprocessing_T6A9FCB5/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)

if hasattr(pre, "feature_names_in_"):
    input_cols = list(pre.feature_names_in_)
else:
    input_cols = feature_cols

X_train = train[input_cols]

selector = FunctionTransformer(func=None, validate=False, feature_names_out="one-to-one")
selector.set_output(transform="pandas")
X_train_pre = pre.transform(X_train)
selector.fit(X_train_pre)

final = Pipeline([("preprocessor", pre), ("selector", selector)])

mlca_runtime.save_model(final, "selector_all.pkl")
print(f"Train shape: {train.shape}, feature cols: {len(feature_cols)}")
print(f"Saved selector_all.pkl with {len(input_cols)} input columns")
