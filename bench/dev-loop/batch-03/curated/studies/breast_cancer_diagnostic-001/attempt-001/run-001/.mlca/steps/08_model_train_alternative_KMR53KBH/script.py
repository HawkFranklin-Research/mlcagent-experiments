import json, pickle
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol["group_column"]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

with open(inputs["07_model_train_logistic_1CFM1FE6/model.pkl"], "rb") as f:
    saved = pickle.load(f)

def find_feat_cols(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        return find_feat_cols(obj.named_steps[list(obj.named_steps)[0]])
    return None

def default_feat_cols():
    drop = set(targets) | set(excluded)
    if group_col:
        drop.add(group_col)
    return [c for c in train.columns if c not in drop]

if isinstance(saved, Pipeline):
    names = list(saved.named_steps)
    feat_cols = find_feat_cols(saved) or default_feat_cols()
    steps = [(n, saved.named_steps[n]) for n in names[:-1]]
    pipe = Pipeline(steps + [("model", HistGradientBoostingClassifier(random_state=42))])
else:
    with open(inputs["05_preprocess_fit_preprocessing_QB1BRK7H/preprocessor.pkl"], "rb") as f:
        pre = pickle.load(f)
    with open(inputs["06_features_select_features_6NMQRFNK/selector.pkl"], "rb") as f:
        sel = pickle.load(f)
    feat_cols = find_feat_cols(pre) or default_feat_cols()
    pipe = Pipeline([("pre", pre), ("sel", sel), ("model", HistGradientBoostingClassifier(random_state=42))])

X_train = train[feat_cols]
y_train = train[targets[0]]
mlca_runtime.fit_record(pipe, X_train, "train", y=y_train)
mlca_runtime.save_model(pipe, "alternative_model.pkl")
print("features:", len(feat_cols), "rows:", len(X_train))
