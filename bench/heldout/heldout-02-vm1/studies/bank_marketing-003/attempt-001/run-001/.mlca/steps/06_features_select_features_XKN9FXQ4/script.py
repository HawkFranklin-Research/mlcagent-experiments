import mlca_runtime
import json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]

train = pd.read_parquet(inputs["train"])
with open(inputs["05_preprocess_fit_preprocessing_CKA3B6GT/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)

feat_cols = list(pre.feature_names_in_)
X = train[feat_cols]
y = pd.factorize(train[targets[0]])[0]

try:
    pre.set_output(transform="pandas")
except Exception:
    pass
n_out = pre.transform(X.head(5)).shape[1]
k = min(20, n_out)

selector = Pipeline([
    ("pre", pre),
    ("select", SelectKBest(score_func=mutual_info_classif, k=k)),
])

mlca_runtime.fit_record(selector, X, "train", y=y)
mlca_runtime.save_model(selector, "selector.pkl")

mlca_runtime.record_metric("n_features_in", float(len(feat_cols)), "train")
mlca_runtime.record_metric("n_features_selected", float(k), "train")
print("features_in:", len(feat_cols))
print("pre_out:", n_out)
print("k:", k)
