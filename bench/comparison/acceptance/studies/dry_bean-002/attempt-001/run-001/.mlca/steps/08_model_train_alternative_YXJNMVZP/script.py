import json, pickle
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())
train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])

targets = protocol["targets"]
excluded = protocol.get("excluded_columns", [])
group = protocol.get("group_column")
target = targets[0]
metric = protocol["task"]["primary_metric"]

def load(p):
    with open(p, "rb") as f:
        return pickle.load(f)

feats = [c for c in train.columns if c not in targets and c not in excluded and c != group]
keylike = []
for c in feats:
    s = train[c]
    if (s.dtype == object or pd.api.types.is_integer_dtype(s)) and s.notna().all() and s.nunique() == len(s):
        keylike.append(c)
feats = [c for c in feats if c not in keylike]

model_obj = load(inp["07_model_train_logistic_FG4AYCRT/model.pkl"])
used = list(model_obj.feature_names_in_) if hasattr(model_obj, "feature_names_in_") else feats
assert target not in used
missing = [c for c in used if c not in train.columns or c not in val.columns]
if missing:
    raise KeyError(f"Missing expected feature columns: {missing}")

if isinstance(model_obj, Pipeline):
    base_steps = model_obj.steps[:-1]
else:
    sel = load(inp["06_features_select_features_H3ASCQ8N/selector.pkl"])
    if isinstance(sel, Pipeline):
        base_steps = sel.steps
    else:
        pre = load(inp["05_preprocess_fit_preprocessing_NBMSGCPN/preprocessor.pkl"])
        base_steps = [("preprocessor", pre), ("selector", sel)]

clf = HistGradientBoostingClassifier(random_state=42, class_weight="balanced")
pipeline = Pipeline(list(base_steps) + [("classifier", clf)])

Xtr = train.loc[:, used]
ytr = train[target]
fitted = mlca_runtime.fit_record(pipeline, Xtr, "train", y=ytr)

classes = list(fitted.classes_)
proba = fitted.predict_proba(val.loc[:, used])
if metric == "roc_auc_ovr":
    score = roc_auc_score(val[target], proba, multi_class="ovr", average="macro", labels=fitted.classes_)
elif metric == "roc_auc":
    positive = classes[1]
    score = roc_auc_score((val[target] == positive).astype(int), proba[:, classes.index(positive)])
else:
    score = float("nan")

mlca_runtime.record_metric(metric, float(score), "val")
mlca_runtime.save_model(fitted, "alternative_model.pkl")
print(len(train), len(val), len(used), round(float(score), 4))
