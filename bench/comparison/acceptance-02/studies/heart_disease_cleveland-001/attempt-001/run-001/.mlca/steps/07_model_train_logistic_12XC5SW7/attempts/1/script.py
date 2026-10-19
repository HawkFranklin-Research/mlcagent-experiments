import json, pickle
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()

proto = json.loads(inp["_protocol"].read_text())
targets = proto["targets"]
excluded = proto["excluded_columns"]
group = proto.get("group_column")
task = proto["task"]
target = targets[0]

train = pd.read_parquet(inp["train"])
val = pd.read_parquet(inp["val"])

pre = pickle.loads(inp["05_preprocess_fit_preprocessing_5G7E554W/preprocessor.pkl"].read_bytes())
sel = pickle.loads(inp["06_features_select_features_Y1ATFPQ6/selector.pkl"].read_bytes())

if hasattr(pre, "feature_names_in_"):
    raw_feats = list(pre.feature_names_in_)
else:
    raw_feats = [c for c in train.columns
                 if c not in targets and c not in excluded and c != group]

Xtr, ytr = train[raw_feats], train[target]
Xva, yva = val[raw_feats], val[target]

# If the saved selector already fronts the raw feature schema, use it alone.
sel_fronts_raw = hasattr(sel, "feature_names_in_") and set(sel.feature_names_in_) == set(raw_feats)

clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
if sel_fronts_raw:
    pipe = Pipeline([("selector", sel), ("clf", clf)])
else:
    pipe = Pipeline([("pre", pre), ("selector", sel), ("clf", clf)])

mlca_runtime.fit_record(pipe, Xtr, "train", y=ytr)

pos = pipe.classes_[1]
proba = pipe.predict_proba(Xva)[:, 1]
auc = float(roc_auc_score((yva == pos).astype(int), proba))
mlca_runtime.record_metric("val_roc_auc", auc, "val")

mlca_runtime.save_model(pipe, "model.pkl")
print(f"train={len(train)} val={len(val)} feats={len(raw_feats)} sel_fronts_raw={sel_fronts_raw}")
print(f"val_roc_auc={auc:.4f}")
