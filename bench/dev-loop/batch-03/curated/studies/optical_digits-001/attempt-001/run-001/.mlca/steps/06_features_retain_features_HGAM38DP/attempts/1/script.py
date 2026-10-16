import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.compose import ColumnTransformer

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])
with open(inputs["05_preprocess_fit_preprocessing_NFXS64E4/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)
manifest = json.loads(inputs["05_preprocess_fit_preprocessing_NFXS64E4/manifest.json"].read_text())

required = ["targets", "excluded_columns", "group_column", "task"]
for key in required:
    if key not in protocol:
        raise KeyError(f"Missing key '{key}' in _protocol. Available keys: {list(protocol.keys())}")

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol["group_column"]
task = protocol["task"]

def get_feature_columns(df, targets, excluded, group):
    cols = [c for c in df.columns if c not in targets and c not in excluded and c != group]
    feature_cols = []
    for c in cols:
        s = df[c]
        if s.dtype == 'object' or isinstance(s.dtype, pd.CategoricalDtype):
            if s.nunique() == len(s):
                continue
        elif pd.api.types.is_integer_dtype(s.dtype):
            if s.nunique() == len(s):
                continue
        feature_cols.append(c)
    return feature_cols

feature_cols = get_feature_columns(train, targets, excluded, group)
X = train[feature_cols]

selector = ColumnTransformer(transformers=[], remainder='passthrough')
selector.set_output(transform='pandas')
selector = mlca_runtime.fit_record(selector, X, partition="train")
mlca_runtime.save_model(selector, "selector_all.pkl")

print(f"Train shape: {train.shape}")
print(f"Feature columns: {len(feature_cols)}")
print("Selector fitted and saved as selector_all.pkl")
