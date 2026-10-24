import mlca_runtime
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
targets = protocol['targets']

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

logit_model = joblib.load(inputs['08_model_train_logistic_YSEARS1W/model.pkl'])

if isinstance(logit_model, Pipeline):
    steps = logit_model.steps[:-1]
    try:
        feat_cols = list(logit_model.feature_names_in_)
    except AttributeError:
        pre = joblib.load(inputs['06_preprocess_fit_preprocessing_VD4Z4Z0N/preprocessor.pkl'])
        feat_cols = list(pre.feature_names_in_)
else:
    pre = joblib.load(inputs['06_preprocess_fit_preprocessing_VD4Z4Z0N/preprocessor.pkl'])
    sel = joblib.load(inputs['07_features_select_features_AKBM21YH/selector.pkl'])
    steps = [('preprocessor', pre), ('selector', sel)]
    feat_cols = list(pre.feature_names_in_)

clf = HistGradientBoostingClassifier(random_state=42)
pipe = Pipeline(steps + [('classifier', clf)])

X_train = train[feat_cols]
y_train = train[targets[0]]
fitted = mlca_runtime.fit_record(pipe, X_train, 'train', y=y_train)

mlca_runtime.save_model(fitted, 'alternative_model.pkl')

X_val = val[feat_cols]
y_val = val[targets[0]]
proba = fitted.predict_proba(X_val)
classes = list(fitted.classes_)
if len(classes) == 2:
    auc = roc_auc_score((y_val == classes[1]).astype(int), proba[:, 1])
    mlca_runtime.record_metric('alternative_val_roc_auc', float(auc), 'val')
    print('val roc_auc:', round(float(auc), 4))
print('train shape:', train.shape, 'features:', len(feat_cols))
print('saved alternative_model.pkl')
