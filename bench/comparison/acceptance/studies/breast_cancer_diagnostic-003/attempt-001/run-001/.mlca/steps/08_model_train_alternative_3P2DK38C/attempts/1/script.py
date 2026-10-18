import json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import mlca_runtime

mlca_runtime.set_seed(42)
seed = 42
paths = mlca_runtime.inputs()
proto = json.loads(paths["_protocol"].read_text())

targets = proto["targets"]
excluded = proto["excluded_columns"]
group = proto.get("group_column")
primary_metric = proto["task"]["primary_metric"]
target = targets[0]

train = pd.read_parquet(paths["train"])
val = pd.read_parquet(paths["val"])

drop = set(targets) | set(excluded) | ({group} if group else set())
feature_cols = []
for c in train.columns:
    if c in drop:
        continue
    s = train[c]
    nun = s.nunique(dropna=True)
    if nun == len(train) and nun > 0:
        if s.dtype == object or pd.api.types.is_integer_dtype(s.dtype):
            continue
    feature_cols.append(c)

X_tr, X_va = train[feature_cols], val[feature_cols]
y_tr, y_va = train[target], val[target]

with open(paths["05_preprocess_fit_preprocessing_5QSFC5J4/preprocessor.pkl"], "rb") as f:
    preprocessor = pickle.load(f)
with open(paths["06_features_select_features_3FVCT0MK/selector.pkl"], "rb") as f:
    selector = pickle.load(f)

if isinstance(selector, Pipeline) and len(selector.named_steps) > 1:
    frontend = selector
else:
    frontend = Pipeline([("pre", preprocessor), ("sel", selector)])

model = Pipeline([
    ("features", frontend),
    ("clf", HistGradientBoostingClassifier(random_state=seed)),
])
mlca_runtime.fit_record(model, X_tr, "train", y_tr)

classes = list(model.named_steps["clf"].classes_)
pos_label = proto["task"].get("positive_label", classes[1])
proba = model.predict_proba(X_va)
col = classes.index(pos_label)
y_true = (y_va == pos_label).astype(int)
auc = roc_auc_score(y_true, proba[:, col])
mlca_runtime.record_metric(primary_metric, float(auc), "val")

mlca_runtime.save_model(model, "alternative_model.pkl")
print("features", len(feature_cols))
print("train", X_tr.shape, "val", X_va.shape)
print(primary_metric, round(float(auc), 4))
