import mlca_runtime

import json
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
protocol = json.loads(inputs["_protocol"].read_text())

task = protocol.get("task", {})
outcome_cols = list(task.get("targets", []))
if not outcome_cols and "columns" in protocol:
    outcome_cols = [protocol["columns"]["target"]] if protocol["columns"].get("target") else []
excluded_cols = list(protocol.get("excluded_columns", []))

outcome_set = set(outcome_cols)
excluded_set = set(excluded_cols)
feature_cols = [c for c in train.columns if c not in outcome_set and c not in excluded_set]

X_train = train[feature_cols]
y_train = train[outcome_cols] if outcome_cols else None
X_val = val[feature_cols]

numeric_cols = X_train.select_dtypes(include="number").columns.tolist()
categorical_cols = [c for c in feature_cols if c not in set(numeric_cols)]

numeric_pipe = Pipeline(
    [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
)
categorical_pipe = Pipeline(
    [
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ],
    remainder="drop",
    sparse_threshold=0,
)

preprocessor = mlca_runtime.fit_record(preprocessor, X_train, "train", y=y_train)

X_val_t = preprocessor.transform(X_val)

mlca_runtime.record_metric("n_features", float(len(feature_cols)), "train")

print(f"train {X_train.shape[0]}x{X_train.shape[1]}")
print(f"val {X_val.shape[0]}x{X_val.shape[1]}")
print(f"numeric {len(numeric_cols)} categorical {len(categorical_cols)}")
print(f"val_transformed {X_val_t.shape[0]}x{X_val_t.shape[1]}")

mlca_runtime.save_model(preprocessor, "preprocessor.pkl")
