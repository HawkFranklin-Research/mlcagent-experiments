import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
with open(inputs["05_preprocess_fit_preprocessing_NVHP5K61/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)
manifest = json.loads(inputs["05_preprocess_fit_preprocessing_NVHP5K61/manifest.json"].read_text())

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol["group_column"]

if hasattr(preprocessor, "feature_names_in_"):
    raw_feature_cols = list(preprocessor.feature_names_in_)
elif isinstance(preprocessor, Pipeline) and hasattr(preprocessor.steps[0][1], "feature_names_in_"):
    raw_feature_cols = list(preprocessor.steps[0][1].feature_names_in_)
else:
    raw_feature_cols = [c for c in train.columns if c not in targets + excluded + [group]]
    for c in raw_feature_cols[:]:
        if train[c].dtype == "object" or pd.api.types.is_integer_dtype(train[c]):
            if train[c].nunique() == len(train):
                raw_feature_cols.remove(c)

X_train_raw = train[raw_feature_cols].copy()

if hasattr(preprocessor, "set_output"):
    preprocessor.set_output(transform="pandas")
X_train_trans = preprocessor.transform(X_train_raw)
if not isinstance(X_train_trans, pd.DataFrame):
    if hasattr(preprocessor, "get_feature_names_out"):
        cols = preprocessor.get_feature_names_out()
    else:
        cols = [f"f{i}" for i in range(X_train_trans.shape[1])]
    X_train_trans = pd.DataFrame(X_train_trans, columns=cols, index=X_train_raw.index)

selector = ColumnTransformer(
    transformers=[("passthrough", "passthrough", list(X_train_trans.columns))],
    remainder="drop"
)
selector.set_output(transform="pandas")
selector = mlca_runtime.fit_record(selector, X_train_trans, partition="train")

mlca_runtime.save_model(selector, "selector_all.pkl")

print(f"Train shape: {train.shape}")
print(f"Raw features: {len(raw_feature_cols)}")
print(f"Transformed features: {X_train_trans.shape[1]}")
