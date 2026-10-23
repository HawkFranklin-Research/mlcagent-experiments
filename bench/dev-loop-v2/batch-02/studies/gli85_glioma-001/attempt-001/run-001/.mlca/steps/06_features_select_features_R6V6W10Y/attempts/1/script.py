import json
import pickle
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
train = pd.read_parquet(inputs["train"])
with open(inputs["05_preprocess_fit_preprocessing_6FVQ17HJ/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol.get("group_column")

def is_key_like(s):
    if pd.api.types.is_object_dtype(s) or isinstance(s.dtype, pd.StringDtype) or pd.api.types.is_categorical_dtype(s):
        return s.nunique(dropna=True) == len(s)
    return False

feature_cols = [
    c for c in train.columns
    if c not in targets and c not in excluded and c != group and not is_key_like(train[c])
]

if hasattr(pre, "feature_names_in_"):
    feature_cols = [c for c in list(pre.feature_names_in_) if c in train.columns]

X = train[feature_cols]
y = train[targets[0]]

if isinstance(pre, Pipeline):
    base_steps = [(name, step) for name, step in pre.steps if not isinstance(step, SelectKBest)]
else:
    base_steps = [("preprocess", pre)]

k = min(20, len(feature_cols))
pipe = Pipeline(base_steps + [
    ("select", SelectKBest(score_func=mutual_info_classif, k=k))
])
fitted = mlca_runtime.fit_record(pipe, X, "train", y)
n_sel = int(fitted.named_steps["select"].get_support().sum())

print(f"train rows: {len(train)}")
print(f"feature columns: {X.shape[1]}")
print(f"selected: {n_sel}")
mlca_runtime.record_metric("n_features_selected", float(n_sel), "train")
mlca_runtime.save_model(fitted, "selector.pkl")
