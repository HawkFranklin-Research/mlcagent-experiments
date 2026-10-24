import mlca_runtime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

seed = 42
mlca_runtime.set_seed(seed)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
target_col = protocol['targets'][0]
excluded = set(protocol.get('excluded_columns', []))
group_col = protocol.get('group_column')

train = pd.read_parquet(inputs['train'])

with open(inputs['07_model_train_logistic_6PX29S2M/model.pkl'], 'rb') as f:
    existing = pickle.load(f)

if isinstance(existing, Pipeline):
    pipe = clone(existing)
    pipe.steps[-1] = ('clf', HistGradientBoostingClassifier(random_state=seed))
    feature_cols = list(pipe.feature_names_in_) if hasattr(pipe, 'feature_names_in_') else None
else:
    with open(inputs['05_preprocess_fit_preprocessing_K9GV8476/preprocessor.pkl'], 'rb') as f:
        pre = pickle.load(f)
    with open(inputs['06_features_select_features_BXHGCTQP/selector.pkl'], 'rb') as f:
        sel = pickle.load(f)
    pipe = Pipeline([
        ('preprocessor', pre),
        ('selector', sel),
        ('clf', HistGradientBoostingClassifier(random_state=seed))
    ])
    feature_cols = list(pre.feature_names_in_) if hasattr(pre, 'feature_names_in_') else None

if feature_cols is None:
    all_cols = [c for c in train.columns if c not in [target_col, group_col] and c not in excluded]
    feature_cols = []
    n = len(train)
    for c in all_cols:
        s = train[c]
        if s.dtype == 'object' or str(s.dtype).startswith('string'):
            if s.nunique() == n:
                continue
        if np.issubdtype(s.dtype, np.integer) or np.issubdtype(s.dtype, np.number):
            if s.nunique() == n:
                continue
        feature_cols.append(c)

missing = [c for c in feature_cols if c not in train.columns]
if missing:
    raise ValueError(f"Missing feature columns: {missing}")

X_train = train[feature_cols].copy()
y_train = train[target_col].copy()

if y_train.nunique() == 2:
    counts = y_train.value_counts(normalize=True)
    if counts.min() < 0.1:
        pipe.set_params(clf__class_weight='balanced')

fitted = mlca_runtime.fit_record(pipe, X_train, 'train', y_train)

if hasattr(fitted, 'predict_proba') and y_train.nunique() == 2:
    proba = fitted.predict_proba(X_train)
    classes = fitted.classes_ if hasattr(fitted, 'classes_') else fitted[-1].classes_
    pos_label = classes[1]
    y_bin = (y_train == pos_label).astype(int)
    auc = roc_auc_score(y_bin, proba[:, 1])
    mlca_runtime.record_metric('train_roc_auc', float(auc), 'train')
    print(f"Train ROC AUC: {auc:.4f}")

mlca_runtime.save_model(fitted, 'alternative_model.pkl')
print(f"Train shape: {X_train.shape}")
print("Saved alternative_model.pkl")
