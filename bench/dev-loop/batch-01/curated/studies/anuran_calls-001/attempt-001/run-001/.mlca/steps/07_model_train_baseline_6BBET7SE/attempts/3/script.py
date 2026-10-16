import mlca_runtime
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs["_protocol"].read_text())
if "targets" not in proto:
    raise KeyError(f"Missing 'targets'. Available: {list(proto.keys())}")
targets = proto["targets"]
excluded = proto.get("excluded_columns", [])
group = proto.get("group_column", None)

train = pd.read_parquet(inputs["train"])
val = pd.read_parquet(inputs["val"])

all_cols = list(train.columns)
feature_cols = [c for c in all_cols if c not in targets and c not in excluded and c != group]
drop_cols = []
for c in feature_cols:
    s = train[c]
    if s.dtype == object or str(s.dtype) in ('string', 'category'):
        if s.nunique() == len(train):
            drop_cols.append(c)
    elif pd.api.types.is_integer_dtype(s):
        if s.nunique() == len(train):
            drop_cols.append(c)
feature_cols = [c for c in feature_cols if c not in drop_cols]

X_train = train[feature_cols].copy()
y_train = train[targets].copy()
X_val = val[feature_cols].copy()
y_val = val[targets].copy()

pre = joblib.load(inputs["05_preprocess_fit_preprocessing_NVHP5K61/preprocessor.pkl"])
sel = joblib.load(inputs["06_features_retain_features_60ES4TDC/selector_all.pkl"])

if isinstance(sel, Pipeline) and ('preprocessor' in sel.named_steps or (len(sel.steps) > 0 and (isinstance(sel.steps[0][1], ColumnTransformer) or isinstance(sel.steps[0][1], Pipeline)))):
    base = sel
else:
    base = Pipeline([('preprocessor', pre), ('selector', sel)])

clf = MultiOutputClassifier(LogisticRegression(max_iter=2000))
model = Pipeline(base.steps + [('clf', clf)])
model.set_output(transform='pandas')

model = mlca_runtime.fit_record(model, X_train, "train", y=y_train)
mlca_runtime.save_model(model, "model.pkl")

pred = model.predict(X_val)
per_target = [f1_score(y_val[col], pred[:, i], average='macro') for i, col in enumerate(targets)]
mean_f1 = float(np.mean(per_target))
mlca_runtime.record_metric("macro_f1", mean_f1, "val")

print(f"Train shape: {train.shape}, Val shape: {val.shape}")
print(f"Targets: {targets}")
print(f"Val macro_f1: {mean_f1:.4f}")
