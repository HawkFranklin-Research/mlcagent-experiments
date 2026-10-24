import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

seed = 42
mlca_runtime.set_seed(seed)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
target_col = protocol['targets'][0]

train = pd.read_parquet(inputs['train'])

with open(inputs['05_preprocess_fit_preprocessing_K9GV8476/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)
with open(inputs['06_features_select_features_BXHGCTQP/selector.pkl'], 'rb') as f:
    sel = pickle.load(f)

# Raw input columns expected by the fitted preprocessor - use exactly these
if hasattr(pre, 'feature_names_in_') and pre.feature_names_in_ is not None:
    feature_cols = list(pre.feature_names_in_)
else:
    feature_cols = [c for c in train.columns if c != target_col]

missing = [c for c in feature_cols if c not in train.columns]
if missing:
    raise ValueError(f"Missing feature columns: {missing[:5]} (total {len(missing)})")

X_train = train[feature_cols].copy()
y_train = train[target_col].copy()

clf = HistGradientBoostingClassifier(random_state=seed)
if y_train.nunique() == 2:
    counts = y_train.value_counts(normalize=True)
    if counts.min() < 0.1:
        clf.set_params(class_weight='balanced')

pipe = Pipeline([
    ('preprocessor', pre),
    ('selector', sel),
    ('clf', clf),
])

fitted = mlca_runtime.fit_record(pipe, X_train, 'train', y_train)

if hasattr(fitted, 'predict_proba') and y_train.nunique() == 2:
    proba = fitted.predict_proba(X_train)
    auc = roc_auc_score((y_train == y_train.unique()[1]).astype(int), proba[:, 1])
    mlca_runtime.record_metric('train_roc_auc', float(auc), 'train')

mlca_runtime.save_model(fitted, 'alternative_model.pkl')
print(f"Train shape: {X_train.shape}")
print("Saved alternative_model.pkl")
