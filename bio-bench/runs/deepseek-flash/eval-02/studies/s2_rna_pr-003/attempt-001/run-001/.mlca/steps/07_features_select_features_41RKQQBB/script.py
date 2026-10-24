import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

targets = protocol["targets"]
excluded = set(protocol["excluded_columns"])
group_col = protocol.get("group_column")
group_cols = {group_col} if group_col else set()
_ = protocol["task"]["primary_metric"]

train = pd.read_parquet(inputs["train"])
target_col = targets[0] if isinstance(targets, list) else targets

candidate = []
for col in train.columns:
    if col in excluded or col in targets or col in group_cols:
        continue
    s = train[col]
    if s.dtype == "object" or str(s.dtype).startswith("string") or s.dtype.name == "category":
        if s.nunique(dropna=True) == len(train):
            continue
    elif pd.api.types.is_integer_dtype(s.dtype):
        if s.nunique(dropna=True) == len(train):
            continue
    candidate.append(col)

with open(inputs["06_preprocess_fit_preprocessing_1X70GNWA/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)

if hasattr(pre, "feature_names_in_"):
    expected = list(pre.feature_names_in_)
    forbidden = set(targets) | excluded | group_cols
    bad = [c for c in expected if c in forbidden]
    if bad:
        raise ValueError(f"Preprocessor expects forbidden columns: {bad}")
    feature_cols = expected
else:
    feature_cols = candidate

X_train = train[feature_cols]
y_train = train[target_col]

Xt = pre.transform(X_train)
n_features = Xt.shape[1] if hasattr(Xt, "shape") else len(Xt[0])
k = min(20, n_features)

selector = Pipeline([
    ("preprocessor", pre),
    ("select", SelectKBest(score_func=mutual_info_classif, k=k)),
])

mlca_runtime.fit_record(selector, X_train, "train", y=y_train)
mlca_runtime.save_model(selector, "selector.pkl")

print(f"train rows: {len(train)}")
print(f"feature columns: {len(feature_cols)}")
print(f"preprocessed features: {n_features}, selected k: {k}")
print("saved selector.pkl")
