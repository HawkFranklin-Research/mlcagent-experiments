import json, pickle
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
excluded = protocol.get("excluded_columns", [])
group_col = protocol.get("group_column")
task = protocol["task"]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

with open(inputs["05_preprocess_fit_preprocessing_F5B2KN9D/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inputs["06_features_select_features_Y4ZJEG70/selector.pkl"], "rb") as f:
    sel = pickle.load(f)


def first_input_names(obj):
    if hasattr(obj, "feature_names_in_"):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        for s in obj.named_steps.values():
            n = first_input_names(s)
            if n:
                return n
    return []


raw_cols = first_input_names(pre)
if not raw_cols:
    raw_cols = [c for c in train.columns
                if c not in targets and c not in excluded and c != group_col]

sel_in = first_input_names(sel)
if raw_cols and sel_in and set(sel_in) == set(raw_cols):
    frontend = sel
else:
    frontend = Pipeline([("pre", pre), ("sel", sel)])

clf = HistGradientBoostingClassifier(random_state=42)
full = Pipeline([("frontend", frontend), ("clf", clf)])

X_train = train[raw_cols]
y_train = train[targets[0]]
fitted = mlca_runtime.fit_record(full, X_train, "train", y=y_train)
mlca_runtime.save_model(fitted, "alternative_model.pkl")

X_val = val[raw_cols]
y_val = val[targets[0]]
proba = fitted.predict_proba(X_val)
classes = fitted.named_steps["clf"].classes_
y_bin = (y_val == classes[1]).astype(int)
auc = roc_auc_score(y_bin, proba[:, 1])
mlca_runtime.record_metric("roc_auc", float(auc), "val")
print("train", X_train.shape, "val_auc", round(float(auc), 4))
