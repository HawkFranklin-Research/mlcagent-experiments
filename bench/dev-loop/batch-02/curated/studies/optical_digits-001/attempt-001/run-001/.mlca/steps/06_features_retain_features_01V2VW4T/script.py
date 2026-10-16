import mlca_runtime
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol.get("targets", [])
excluded = protocol.get("excluded_columns", [])
group = protocol.get("group_column", None)

train = pd.read_parquet(inputs["train"])

with open(inputs["05_preprocess_fit_preprocessing_78KYAZ91/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)

if hasattr(preprocessor, "set_output"):
    preprocessor.set_output(transform="pandas")

if hasattr(preprocessor, "feature_names_in_"):
    raw_cols = list(preprocessor.feature_names_in_)
else:
    raw_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group]

X_train_raw = train[raw_cols]
X_train_pre = preprocessor.transform(X_train_raw)
if not isinstance(X_train_pre, pd.DataFrame):
    X_train_pre = pd.DataFrame(
        X_train_pre,
        columns=preprocessor.get_feature_names_out(),
        index=X_train_raw.index,
    )

preprocessed_cols = list(X_train_pre.columns)
selector = ColumnTransformer(
    transformers=[("passthrough", "passthrough", preprocessed_cols)],
    remainder="drop",
)
selector.set_output(transform="pandas")
selector = mlca_runtime.fit_record(selector, X_train_pre, "train")

mlca_runtime.save_model(selector, "selector_all.pkl")

print(f"Features: {len(preprocessed_cols)}")
print("Selector saved.")
