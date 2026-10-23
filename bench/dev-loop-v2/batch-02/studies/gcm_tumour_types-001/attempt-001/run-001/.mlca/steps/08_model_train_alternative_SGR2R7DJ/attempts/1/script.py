import mlca_runtime, json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp["_protocol"].read_text())

targets = protocol["targets"]
excluded = list(protocol.get("excluded_columns", []))
group_col = protocol.get("group_column")
task = protocol["task"]

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])

with open(inp["latest_model"], "rb") as f:
    base = pickle.load(f)

if not isinstance(base, Pipeline):
    raise RuntimeError("latest_model is not a Pipeline")

if hasattr(base, "feature_names_in_"):
    feature_cols = list(base.feature_names_in_)
else:
    feature_cols = [c for c in train.columns
                    if c not in targets and c not in excluded
                    and (group_col is None or c != group_col)]

X_train = train[feature_cols]
y_train = train[targets[0]]

clf = HistGradientBoostingClassifier(random_state=42, class_weight="balanced")
pipe = Pipeline(list(base.steps[:-1]) + [("clf", clf)])

mlca_runtime.fit_record(pipe, X_train, "train", y=y_train)
mlca_runtime.save_model(pipe, "alternative_model.pkl")

y_val = val[targets[0]]
proba = pipe.predict_proba(val[feature_cols])
try:
    auc = roc_auc_score(y_val, proba, multi_class="ovr", average="macro",
                        labels=pipe.classes_)
    mlca_runtime.record_metric("roc_auc_ovr", float(auc), "val")
    print(f"val roc_auc_ovr={auc:.4f}")
except ValueError:
    print("val auc skipped (class coverage)")

print(f"train={X_train.shape} features={len(feature_cols)} classes={len(pipe.classes_)}")
