import json, pickle
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

SEED = 42
mlca_runtime.set_seed(SEED)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol["targets"]
excluded = list(protocol.get("excluded_columns") or [])
group = protocol.get("group_column")

train = pd.read_parquet(inputs["train"])
y_train = train[targets[0]].copy()

def keylike(s):
    if pd.api.types.is_integer_dtype(s.dtype) or s.dtype == object or pd.api.types.is_string_dtype(s.dtype):
        n = s.nunique(dropna=True)
        if n == len(s) and n > 1:
            return True
    return False

feature_cols = [c for c in train.columns
                if c not in targets and c not in excluded
                and c != group and not keylike(train[c])]
X_train = train[feature_cols].copy()

with open(inputs["05_preprocess_fit_preprocessing_H4RCZJ73/preprocessor.pkl"], "rb") as f:
    pre = pickle.load(f)
with open(inputs["06_features_select_features_F6NPFNPS/selector.pkl"], "rb") as f:
    sel = pickle.load(f)

def accepts_raw(obj):
    fi = getattr(obj, "feature_names_in_", None)
    return fi is not None and list(fi) == feature_cols

clf = HistGradientBoostingClassifier(random_state=SEED)
if accepts_raw(sel):
    pipe = Pipeline([("selector", sel), ("clf", clf)])
elif accepts_raw(pre):
    pipe = Pipeline([("preprocessor", pre), ("selector", sel), ("clf", clf)])
else:
    pipe = Pipeline([("preprocessor", pre), ("selector", sel), ("clf", clf)])

pipe = mlca_runtime.fit_record(pipe, X_train, "train", y=y_train)
mlca_runtime.save_model(pipe, "alternative_model.pkl")

try:
    val = pd.read_parquet(inputs["val"])
    X_val = val[feature_cols].copy()
    y_val = val[targets[0]]
    proba = pipe.predict_proba(X_val)
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(y_val, proba, multi_class="ovr",
                              average="macro", labels=pipe.classes_))
    mlca_runtime.record_metric("roc_auc_ovr", auc, "val")
    print(f"val roc_auc_ovr={auc:.4f}")
except Exception as e:
    print(f"val skipped: {type(e).__name__}")

print(f"features={len(feature_cols)} train_rows={len(X_train)}")
