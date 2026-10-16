import json
import pickle
import mlca_runtime
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group_col = protocol.get("group_column")

train = pd.read_parquet(inputs["train"])

with open(inputs["05_preprocess_fit_preprocessing_CZ0KAB5J/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inputs["06_features_select_features_X4RMHKJ0/selector.pkl"], "rb") as f:
    sel = pickle.load(f)

def has_pre(obj):
    if not isinstance(obj, Pipeline):
        return False
    first = list(obj.named_steps.values())[0]
    return isinstance(first, ColumnTransformer)

front = sel if has_pre(sel) else Pipeline([("pre", pre), ("sel", sel)])

if hasattr(front, "feature_names_in_"):
    feat_cols = list(front.feature_names_in_)
else:
    drop = set(targets) | set(excluded)
    if group_col:
        drop.add(group_col)
    feat_cols = [c for c in train.columns if c not in drop]

X = train[feat_cols]
y = train[targets[0]]

counts = y.value_counts()
kw = {"class_weight": "balanced"} if counts.min() / counts.sum() < 0.1 else {}
model = HistGradientBoostingClassifier(random_state=42, **kw)

pipe = Pipeline([("front", front), ("clf", model)])
fitted = mlca_runtime.fit_record(pipe, X, "train", y)

mlca_runtime.save_model(fitted, "alternative_model.pkl")

classes = list(fitted.named_steps["clf"].classes_)
proba = fitted.predict_proba(X)[:, 1]
mlca_runtime.record_metric("train_roc_auc", float(roc_auc_score(y, proba, pos_label=classes[1])), "train")
print("saved alternative_model.pkl", X.shape)
