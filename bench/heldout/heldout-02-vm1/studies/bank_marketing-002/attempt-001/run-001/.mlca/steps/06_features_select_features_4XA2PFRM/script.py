import json
import pickle
import mlca_runtime
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
train = pd.read_parquet(paths["train"])
protocol = json.loads(paths["_protocol"].read_text())
with open(paths["05_preprocess_fit_preprocessing_VSWY5WH2/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)

targets = list(protocol["targets"])
target = targets[0]
excluded = set(protocol["excluded_columns"])
group = protocol.get("group_column")
drop = set(targets) | excluded | ({group} if group else set())

if hasattr(pre, "feature_names_in_"):
    feature_cols = list(pre.feature_names_in_)
else:
    feature_cols = [c for c in train.columns if c not in drop]

y_train = train[target].astype("category").cat.codes

if isinstance(pre, Pipeline) and pre.steps and isinstance(pre.steps[-1][1], SelectKBest):
    pipeline = pre
    k_value = pre.steps[-1][1].k
else:
    try:
        n_out = len(pre.get_feature_names_out())
    except Exception:
        n_out = len(feature_cols)
    k_value = max(1, min(20, n_out))
    selector = SelectKBest(score_func=mutual_info_classif, k=k_value)
    if isinstance(pre, Pipeline):
        name = "selector" if "selector" not in pre.named_steps else "selector_final"
        pipeline = Pipeline(list(pre.steps) + [(name, selector)])
    else:
        pipeline = Pipeline([("preprocess", pre), ("selector", selector)])

X_train = train.loc[:, feature_cols]
fitted = mlca_runtime.fit_record(pipeline, X_train, "train", y=y_train)

selected = None
for _, step in getattr(fitted, "steps", []):
    if hasattr(step, "get_support"):
        selected = int(step.get_support().sum())
if selected is None:
    selected = len(feature_cols)

mlca_runtime.record_metric("n_input_features", float(len(feature_cols)), "train")
mlca_runtime.record_metric("n_selected_features", float(selected), "train")
mlca_runtime.save_model(fitted, "selector.pkl")
print(f"input_features={len(feature_cols)} k={k_value} selected={selected}")
