import json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import mlca_runtime

seed = 42
mlca_runtime.set_seed(seed)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs['_protocol'].read_text())
targets = list(protocol['targets'])
excluded = list(protocol.get('excluded_columns', []))
group = protocol.get('group_column')
task = protocol['task']
metric = task['primary_metric']

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

prev = pickle.loads(inputs['07_model_train_logistic_YWHJPFAZ/model.pkl'].read_bytes())

if isinstance(prev, Pipeline) and hasattr(prev, 'feature_names_in_'):
    feature_cols = list(prev.feature_names_in_)
else:
    def is_key(c):
        s = train[c]; n = s.nunique()
        if n != len(s): return False
        if pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s):
            return (s.max() - s.min()) == (len(s) - 1)
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            return True
        return False
    drop = set(targets) | set(excluded) | ({group} if group else set())
    feature_cols = [c for c in train.columns if c not in drop and not is_key(c)]

if isinstance(prev, Pipeline):
    base_steps = prev.steps[:-1]
else:
    pre = pickle.loads(inputs['05_preprocess_fit_preprocessing_A5VJ9RPH/preprocessor.pkl'].read_bytes())
    sel = pickle.loads(inputs['06_features_select_features_KR1QEBV4/selector.pkl'].read_bytes())
    base_steps = [('preprocessor', pre), ('selector', sel)]

target = targets[0]
X, y = train[feature_cols], train[target]
X_val, y_val = val[feature_cols], val[target]

kw = dict(random_state=seed)
if y.value_counts(normalize=True).min() < 0.1:
    kw['class_weight'] = 'balanced'
new_est = HistGradientBoostingClassifier(**kw)

names = [n for n, _ in base_steps]
clf_name = 'clf' if 'clf' not in names else 'classifier'
pipeline = Pipeline(base_steps + [(clf_name, new_est)])

fitted = mlca_runtime.fit_record(pipeline, X, 'train', y=y)

proba = fitted.predict_proba(X_val)
classes = fitted.named_steps[clf_name].classes_
auc = roc_auc_score(y_val, proba, multi_class='ovr', average='macro', labels=classes)
mlca_runtime.record_metric(metric, float(auc), 'val')

mlca_runtime.save_model(fitted, 'alternative_model.pkl')
print('features', len(feature_cols), 'train', len(X), 'val', len(X_val))
print('val', metric, round(float(auc), 4))
