import json
import pickle
import pandas as pd
import mlca_runtime
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Set seed for reproducibility
mlca_runtime.set_seed(42)

# Read inputs
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

# Load data
train_df = pd.read_parquet(inputs["train"])
val_df = pd.read_parquet(inputs["val"])

# Load preprocessor and selector
with open(inputs["05_preprocess_fit_preprocessing_S3GJVMHP/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)

with open(inputs["06_features_select_features_AX4GAMWK/selector.pkl"], "rb") as f:
    selector = pickle.load(f)

# Extract target column from protocol
target_col = protocol["task"]["targets"][0]  # "heart_disease_binary"

# Separate features and target
X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col]
X_val = val_df.drop(columns=[target_col])
y_val = val_df[target_col]

# Build pipeline: preprocessing + feature selection + classifier
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("selector", selector),
    ("classifier", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)),
])

# Fit on training data only
fitted_pipeline = mlca_runtime.fit_record(pipeline, X_train, "train", y_train)

# Save the complete pipeline
mlca_runtime.save_model(fitted_pipeline, "model.pkl")

# Print summary
print(f"Train shape: {train_df.shape}, Val shape: {val_df.shape}")
print(f"Target: {target_col}, Task: {protocol['task']['type']}")
print("Pipeline fitted and saved as model.pkl")
