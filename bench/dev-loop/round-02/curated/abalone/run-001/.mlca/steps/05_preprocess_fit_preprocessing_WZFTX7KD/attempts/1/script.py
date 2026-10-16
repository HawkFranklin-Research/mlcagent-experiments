import mlca_runtime
import pandas as pd
import json
import pickle
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)

# Load inputs
inputs = mlca_runtime.inputs()
train_path = inputs["train"]
val_path = inputs["val"]
protocol_path = inputs["_protocol"]

# Read data
train_df = pd.read_parquet(train_path)
val_df = pd.read_parquet(val_path)
with open(protocol_path, "r") as f:
    protocol = json.load(f)

# Extract protocol info
target_col = protocol["task"]["targets"][0]  # "Rings"
excluded_cols = protocol.get("excluded_columns", [])
task_type = protocol["task"]["type"]

print(f"Train shape: {train_df.shape}")
print(f"Val shape: {val_df.shape}")
print(f"Target: {target_col}")
print(f"Task: {task_type}")

# Identify feature columns (exclude target and any excluded columns)
feature_cols = [c for c in train_df.columns if c != target_col and c not in excluded_cols]
print(f"Feature columns ({len(feature_cols)}): {feature_cols}")

# Split features and target
X_train = train_df[feature_cols]
y_train = train_df[target_col]
X_val = val_df[feature_cols]
y_val = val_df[target_col]

# Identify numeric and categorical columns
numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = X_train.select_dtypes(exclude=["number"]).columns.tolist()

print(f"Numeric cols ({len(numeric_cols)}): {numeric_cols}")
print(f"Categorical cols ({len(categorical_cols)}): {categorical_cols}")

# Build transformers
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ],
    remainder="drop",
    verbose_feature_names_out=False
)

# Fit on training features only
preprocessor.fit(X_train)

# Save preprocessor
output_dir = mlca_runtime.output_dir()
preprocessor_path = output_dir / "preprocessor.pkl"
with open(preprocessor_path, "wb") as f:
    pickle.dump(preprocessor, f)

# Also save via mlca_runtime.save_model for tracking
mlca_runtime.save_model(preprocessor, "preprocessor.pkl")

print(f"Preprocessor saved to {preprocessor_path}")
print(f"Transformed train shape: {preprocessor.transform(X_train).shape}")
print(f"Transformed val shape: {preprocessor.transform(X_val).shape}")
