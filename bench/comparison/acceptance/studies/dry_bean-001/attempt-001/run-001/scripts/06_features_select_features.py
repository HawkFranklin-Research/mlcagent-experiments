import json, pickle
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol.get("group_column")

train = pd.read_parquet(inp["train"])
feature_cols = [c for c in train.columns
                if c not in targets and c not in excluded and c != group_col]

with open(inp["05_preprocess_fit_preprocessing_K97TX9BR/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)

if not isinstance(pre, Pipeline):
    pre = Pipeline([("pre", pre)])
pre.steps = list(pre.steps)
if not any(isinstance(s, SelectKBest) for _, s in pre.steps):
    k = min(20, len(feature_cols))
    pre.steps.append(("select", SelectKBest(mutual_info_classif, k=k)))

selector = pre
X = train[feature_cols]
y = train[targets[0]]
selector = mlca_runtime.fit_record(selector, X, "train", y=y)
mlca_runtime.save_model(selector, "selector.pkl")

last = selector.steps[-1][1]
n_selected = int(last.get_support().sum())
print(f"features in: {len(feature_cols)}; selected: {n_selected}")
mlca_runtime.record_metric("n_features_selected", float(n_selected), "train")
