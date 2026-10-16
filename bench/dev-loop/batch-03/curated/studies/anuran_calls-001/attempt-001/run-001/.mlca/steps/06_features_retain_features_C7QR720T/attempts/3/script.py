import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol["group_column"]
train_df = pd.read_parquet(inputs["train"])
with open(inputs["05_preprocess_fit_preprocessing_WXDY6VPK/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)

if hasattr(preprocessor, "feature_names_in_"):
    feature_cols = list(preprocessor.feature_names_in_)
else:
    all_cols = list(train_df.columns)
    feature_cols = [c for c in all_cols if c not in targets + excluded + [group]]
    key_like = []
    for c in feature_cols:
        if train_df[c].dtype == 'object' or train_df[c].dtype.kind in 'iu':
            if train_df[c].nunique() == len(train_df):
                key_like.append(c)
    feature_cols = [c for c in feature_cols if c not in key_like]

X_train = train_df[feature_cols]
y_train = train_df[targets] if targets else None

selector = FunctionTransformer(func=None, validate=False)
mlca_runtime.fit_record(selector, X_train, partition="train", y=y_train)

if isinstance(preprocessor, Pipeline):
    steps = list(preprocessor.steps) + [('selector', selector)]
    full_pipeline = Pipeline(steps)
else:
    full_pipeline = Pipeline([('preprocessor', preprocessor), ('selector', selector)])

mlca_runtime.save_model(full_pipeline, "selector_all.pkl")
print(f"Features: {len(feature_cols)}")
print(f"Train shape: {X_train.shape}")
print("Saved selector_all.pkl")
