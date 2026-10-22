import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol.get("group_column")
primary = protocol["task"]["primary_metric"]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

with open(inputs["05_preprocess_fit_preprocessing_TZWQXPN7/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)

skip = set(targets) | set(excluded)
if group_col:
    skip.add(group_col)
feature_cols = []
for c in train.columns:
    if c in skip:
        continue
    s = train[c]
    if s.dtype == object and s.nunique(dropna=True) == len(s):
        continue
    if pd.api.types.is_integer_dtype(s) and s.is_unique:
        continue
    feature_cols.append(c)

expected = list(getattr(preprocessor, "feature_names_in_", feature_cols))
missing = [c for c in expected if c not in train.columns]
if missing:
    raise ValueError(f"missing expected feature columns: {missing[:5]}")
X_train = train[expected]
y_train = train[targets[0]]

w = preprocessor.transform(X_train.head(min(20, len(X_train)))).shape[1]
k = max(1, min(20, w))

selector = Pipeline([
    ("prep", preprocessor),
    ("select", SelectKBest(mutual_info_classif, k=k)),
])
mlca_runtime.fit_record(selector, X_train, "train", y_train)
mlca_runtime.save_model(selector, "selector.pkl")

mlca_runtime.record_metric("n_features_input", float(len(expected)), "train")
mlca_runtime.record_metric("n_features_selected", float(k), "train")
print(f"features={len(expected)} transformed={w} k={k}")
print(f"protocol_targets={targets} excluded={len(excluded)}")
