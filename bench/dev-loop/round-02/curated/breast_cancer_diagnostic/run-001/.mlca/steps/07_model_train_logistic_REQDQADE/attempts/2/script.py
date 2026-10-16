import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

train_df = pd.read_parquet(inputs["train"])
val_df = pd.read_parquet(inputs["val"])

target_col = protocol["answers"]["columns"]["target"]
y_train = train_df[target_col]
X_train = train_df.drop(columns=[target_col])
y_val = val_df[target_col]
X_val = val_df.drop(columns=[target_col])

with open(inputs["05_preprocess_fit_preprocessing_B920SHZP/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)

with open(inputs["06_features_select_features_J8ZJM9E3/selector.pkl"], "rb") as f:
    selector = pickle.load(f)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("selector", selector),
    ("classifier", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))
])

fitted_pipeline = mlca_runtime.fit_record(pipeline, X_train, "train", y=y_train)

val_proba = fitted_pipeline.predict_proba(X_val)[:, 1]
from sklearn.metrics import roc_auc_score
val_auc = roc_auc_score(y_val, val_proba)
mlca_runtime.record_metric("roc_auc", float(val_auc), "val")

mlca_runtime.save_model(fitted_pipeline, "model.pkl")
print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")
print(f"Validation ROC AUC: {val_auc:.4f}")
