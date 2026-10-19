import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = list(protocol["targets"])
excluded = list(protocol.get("excluded_columns", []))
group = protocol.get("group_column")
target = targets[0]

train = pd.read_parquet(inputs["train"])
feature_cols = [c for c in train.columns
                if c not in targets and c not in excluded and c != group]
X = train[feature_cols]
y = train[target]

with open(inputs["05_preprocess_fit_preprocessing_Z8NMCAXW/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inputs["06_features_select_features_53HVANE8/selector.pkl"], "rb") as f:
    sel = pickle.load(f)


def raw_names(obj):
    names = list(getattr(obj, "feature_names_in_", []))
    if not names and isinstance(obj, Pipeline):
        for step in obj.named_steps.values():
            n = list(getattr(step, "feature_names_in_", []))
            if n:
                return n
    return names


pre_in = raw_names(pre)
sel_in = raw_names(sel)
cols = set(feature_cols)
sel_raw = bool(sel_in) and set(sel_in) <= cols
pre_raw = bool(pre_in) and set(pre_in) <= cols

if sel_raw:
    frontend = sel
elif pre_raw:
    frontend = Pipeline([("preprocess", pre), ("select", sel)])
else:
    frontend = Pipeline([("preprocess", pre), ("select", sel)])

model = Pipeline([
    ("frontend", frontend),
    ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)),
])

mlca_runtime.fit_record(model, X, "train", y=y)
mlca_runtime.save_model(model, "model.pkl")
print(f"rows={len(X)} feats={len(feature_cols)} classes={y.nunique()} frontend={'selector' if sel_raw else 'pre+sel'}")
