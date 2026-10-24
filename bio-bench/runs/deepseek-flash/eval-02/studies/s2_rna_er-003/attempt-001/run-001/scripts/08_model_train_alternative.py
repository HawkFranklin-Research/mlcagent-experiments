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

sel_names = getattr(sel, 'feature_names_in_', None)
sel_takes_raw = sel_names is not None and set(sel_names).issubset(set(train.columns))

y_train = train[target_col]
clf = HistGradientBoostingClassifier(random_state=seed)
if y_train.nunique() == 2 and y_train.value_counts(normalize=True).min() < 0.1:
    clf.set_params(class_weight='balanced')

if sel_takes_raw:
    feature_cols = list(sel_names)
    pipe = Pipeline([('selector', sel), ('clf', clf)])
else:
    pre_names = getattr(pre, 'feature_names_in_', None)
    feature_cols = list(pre_names) if pre_names is not None else [c for c in train.columns if c != target_col]
    pipe = Pipeline([('preprocessor', pre), ('selector', sel), ('clf', clf)])

missing = [c for c in feature_cols if c not in train.columns]
if missing:
    raise ValueError(f"Missing feature columns: {missing[:5]} (total {len(missing)})")

X_train = train[feature_cols]

fitted = mlca_runtime.fit_record(pipe, X_train, 'train', y_train)

if y_train.nunique() == 2:
    proba = fitted.predict_proba(X_train)
    classes = list(getattr(fitted, 'classes_', fitted.named_steps['clf'].classes_))
    pos = 1 if len(classes) > 1 else 0
    auc = roc_auc_score((y_train == classes[pos]).astype(int), proba[:, pos])
    mlca_runtime.record_metric('train_roc_auc', float(auc), 'train')

mlca_runtime.save_model(fitted, 'alternative_model.pkl')
print(f"Frontend: {'selector-only' if sel_takes_raw else 'pre+selector'}")
print(f"Train shape: {X_train.shape}")
print("Saved alternative_model.pkl")
