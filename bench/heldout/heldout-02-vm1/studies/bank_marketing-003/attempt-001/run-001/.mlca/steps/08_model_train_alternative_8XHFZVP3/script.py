import mlca_runtime
import json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

seed = 42
mlca_runtime.set_seed(seed)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())

targets = protocol["targets"]
excluded = protocol["excluded_columns"]
group = protocol.get("group_column")
target = targets[0]

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

def default_features(df):
    drop = set(targets) | set(excluded)
    if group:
        drop.add(group)
    return [c for c in df.columns if c not in drop]

with open(inputs["07_model_train_logistic_5PDZPGWS/model.pkl"], "rb") as f:
    saved = pickle.load(f)

if isinstance(saved, Pipeline) and hasattr(saved, "feature_names_in_"):
    feat_cols = list(saved.feature_names_in_)
    steps = list(saved.steps)
    last_name = steps[-1][0]
    steps[-1] = (last_name, HistGradientBoostingClassifier(random_state=seed))
    pipe = Pipeline(steps)
else:
    with open(inputs["05_preprocess_fit_preprocessing_CKA3B6GT/preprocessor.pkl"], "rb") as f:
        pre = pickle.load(f)
    with open(inputs["06_features_select_features_XKN9FXQ4/selector.pkl"], "rb") as f:
        sel = pickle.load(f)
    pipe = Pipeline([("pre", pre), ("sel", sel),
                     ("clf", HistGradientBoostingClassifier(random_state=seed))])
    feat_cols = default_features(train)

X_train = train[feat_cols]
y_train = train[target]
X_val = val[feat_cols]
y_val = val[target]

mlca_runtime.fit_record(pipe, X_train, "train", y=y_train)

classes = list(pipe.classes_)
proba = pipe.predict_proba(X_val)
pos_idx = classes.index(classes[-1])
pos_label = classes[-1]
y_bin = (y_val == pos_label).astype(int)
auc = roc_auc_score(y_bin, proba[:, pos_idx])

mlca_runtime.record_metric("val_roc_auc", float(auc), "val")
mlca_runtime.save_model(pipe, "alternative_model.pkl")

print(f"feat={len(feat_cols)} classes={classes} pos={pos_label} val_auc={auc:.4f}")
