import json
import pickle
import mlca_runtime
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

targets = list(protocol["targets"])
excluded = list(protocol.get("excluded_columns", []) or [])
group = protocol.get("group_column")
target = targets[0]

with open(inputs["07_model_train_logistic_H9QQPD1P/model.pkl"], "rb") as f:
    saved = pickle.load(f)

if hasattr(saved, "feature_names_in_"):
    feat_cols = list(saved.feature_names_in_)
elif isinstance(saved, Pipeline) and "preprocess" in saved.named_steps and hasattr(saved.named_steps["preprocess"], "feature_names_in_"):
    feat_cols = list(saved.named_steps["preprocess"].feature_names_in_)
else:
    drop = set(targets) | set(excluded) | ({group} if group else set())
    feat_cols = []
    for c in train.columns:
        if c in drop:
            continue
        s = train[c]
        if s.dtype == object and s.nunique() == len(s):
            continue
        if pd.api.types.is_integer_dtype(s) and s.nunique() == len(s) and s.nunique() > 20:
            continue
        feat_cols.append(c)

y_train = train[target]
minority = float(y_train.value_counts(normalize=True).min())
cw = "balanced" if minority < 0.10 else None

clf = HistGradientBoostingClassifier(random_state=42, class_weight=cw)

if isinstance(saved, Pipeline):
    steps = list(saved.named_steps.items())
    last_name = steps[-1][0]
    pipe = Pipeline(steps[:-1] + [(last_name, clf)])
else:
    with open(inputs["05_preprocess_fit_preprocessing_S0BP8ZJM/preprocessor.pkl"], "rb") as f:
        pre = pickle.load(f)
    with open(inputs["06_features_select_features_R97WA2VA/selector.pkl"], "rb") as f:
        sel = pickle.load(f)
    pipe = Pipeline([("preprocess", pre), ("select", sel), ("clf", clf)])

X_train = train[feat_cols]
pipe = mlca_runtime.fit_record(pipe, X_train, "train", y=y_train)

proba = pipe.predict_proba(val[feat_cols])
classes = list(pipe.classes_)
pos = 1 if len(classes) > 1 else 0
y_val = (val[target] == classes[pos]).astype(int)
mlca_runtime.record_metric("roc_auc", float(roc_auc_score(y_val, proba[:, pos])), "val")

mlca_runtime.save_model(pipe, "alternative_model.pkl")
print(f"rows_train={len(train)} rows_val={len(val)} feats={len(feat_cols)}")
print(f"minority_frac={minority:.3f} class_weight={cw}")
