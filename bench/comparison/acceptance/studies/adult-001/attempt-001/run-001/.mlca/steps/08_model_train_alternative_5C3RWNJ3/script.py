import mlca_runtime
import json, pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
target_col = protocol['targets'][0]

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

with open(inputs['07_model_train_logistic_QY4T2JSR/model.pkl'], 'rb') as f:
    base = pickle.load(f)

if not isinstance(base, Pipeline):
    raise TypeError(f"Expected saved model to be a Pipeline, got {type(base)}")

input_cols = list(base.feature_names_in_) if hasattr(base, 'feature_names_in_') else list(base.steps[0][1].feature_names_in_)
X_train = train[input_cols]
y_train = train[target_col]
X_val = val[input_cols]
y_val = val[target_col]

minority_frac = y_train.value_counts(normalize=True).min()
clf = HistGradientBoostingClassifier(
    random_state=42,
    class_weight='balanced' if minority_frac < 0.1 else None
)

steps = base.steps[:-1] + [('clf', clf)]
pipe = Pipeline(steps)

fitted = mlca_runtime.fit_record(pipe, X_train, 'train', y_train)

if protocol['task']['primary_metric'] == 'roc_auc':
    proba = fitted.predict_proba(X_val)[:, 1]
    classes = fitted.classes_
    pos_label = classes[1]
    y_val_bin = (y_val == pos_label).astype(int)
    auc = roc_auc_score(y_val_bin, proba)
    mlca_runtime.record_metric('roc_auc', float(auc), 'val')
    print(f"val roc_auc: {auc:.4f}")

mlca_runtime.save_model(fitted, 'alternative_model.pkl')
print("Saved alternative_model.pkl")
