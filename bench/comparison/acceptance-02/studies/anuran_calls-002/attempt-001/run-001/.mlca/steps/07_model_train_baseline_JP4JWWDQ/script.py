import mlca_runtime
import json, pickle
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs['_protocol'].read_text())
targets = proto['targets']
excluded = proto.get('excluded_columns', [])
group = proto.get('group_column')

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

feature_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group]
key_cols = [c for c in feature_cols if (train[c].dtype == 'object' or pd.api.types.is_integer_dtype(train[c])) and train[c].nunique(dropna=False) == len(train)]
feature_cols = [c for c in feature_cols if c not in key_cols]

X_train = train[feature_cols].copy()
X_val = val[feature_cols].copy()
y_train = train[targets].copy()
y_val = val[targets].copy()

with open(inputs['05_preprocess_fit_preprocessing_A8X1QVXT/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)
with open(inputs['06_features_retain_features_ZGB5PBQC/selector_all.pkl'], 'rb') as f:
    sel = pickle.load(f)

steps = []
if pre is not None:
    steps.append(('pre', pre))
if sel is not None:
    if not (isinstance(pre, Pipeline) and any(name in pre.named_steps for name in ['selector', 'select', 'feature_selection'])):
        steps.append(('sel', sel))
steps.append(('clf', MultiOutputClassifier(LogisticRegression(max_iter=2000))))
model = Pipeline(steps) if len(steps) > 1 else steps[0][1]

model = mlca_runtime.fit_record(model, X_train, 'train', y=y_train)

pred = model.predict(X_val)
f1s = [f1_score(y_val[t], pred[:, i], average='macro') for i, t in enumerate(targets)]
mlca_runtime.record_metric('macro_f1', float(np.mean(f1s)), 'val')
mlca_runtime.save_model(model, 'model.pkl')
print(f"Train shape: {train.shape}, features: {len(feature_cols)}")
print(f"Targets: {targets}")
print(f"Model: {type(model).__name__}")
