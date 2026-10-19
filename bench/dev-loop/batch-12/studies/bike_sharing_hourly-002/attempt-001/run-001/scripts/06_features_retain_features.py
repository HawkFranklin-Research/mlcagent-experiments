import mlca_runtime, json, pickle
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())
targets = list(proto["targets"])
excluded = list(proto["excluded_columns"])
group = proto.get("group_column")

train = pd.read_parquet(inputs["train"])
with open(inputs["05_preprocess_fit_preprocessing_T5EPBF1H/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)

drop = set(targets) | set(excluded)
if group:
    drop.add(group)

def is_keylike(s):
    if s.isna().any():
        return False
    if s.dtype == object and s.is_unique:
        return True
    if pd.api.types.is_integer_dtype(s) and s.is_unique:
        return True
    return False

feature_cols = [c for c in train.columns if c not in drop and not is_keylike(train[c])]
if hasattr(preprocessor, "feature_names_in_"):
    feature_cols = [c for c in preprocessor.feature_names_in_ if c not in drop]

X = train[feature_cols].copy()
y = train[targets[0]]

try:
    width = preprocessor.transform(X).shape[1]
except Exception:
    width = len(feature_cols)

pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("selector", SelectKBest(k="all", score_func=f_regression)),
])
if width <= 300:
    try:
        pipe.set_output(transform="pandas")
    except Exception:
        pass

mlca_runtime.fit_record(pipe, X, "train", y)
mlca_runtime.save_model(pipe, "selector_all.pkl")

mlca_runtime.record_metric("n_features", float(len(feature_cols)), "train")
mlca_runtime.record_metric("n_train_rows", float(len(X)), "train")
print(f"features={len(feature_cols)} width={width} rows={len(X)}")
