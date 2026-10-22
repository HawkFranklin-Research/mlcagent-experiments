import mlca_runtime
import pandas as pd, pickle, json
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs['_protocol'].read_text())
targets = protocol['targets']
excluded = protocol['excluded_columns']
group_col = protocol.get('group_column')
task = protocol['task']
primary_metric = task['primary_metric']

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

with open(inputs['05_preprocess_fit_preprocessing_2E873D38/preprocessor.pkl'], 'rb') as f:
    preprocessor = pickle.load(f)
with open(inputs['06_features_select_features_CMGHE25G/selector.pkl'], 'rb') as f:
    selector = pickle.load(f)

def raw_features(obj):
    try:
        v = obj.feature_names_in_
    except (AttributeError, ValueError, TypeError):
        return []
    if v is None:
        return []
    try:
        return [str(x) for x in v]
    except TypeError:
        return []

pre_raw = raw_features(preprocessor)
drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)
if pre_raw:
    missing = [c for c in pre_raw if c not in train.columns]
    if missing:
        raise ValueError(f"Missing preprocessor features: {missing}")
    feature_cols = pre_raw
else:
    feature_cols = [c for c in train.columns if c not in drop]

target = targets[0]
X = train[feature_cols]
y = train[target]

clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000)

# RL-0011: use the most complete saved pipeline (selector may wrap preprocessing).
if isinstance(selector, Pipeline) and len(selector.steps) > 1:
    pipeline = Pipeline(list(selector.steps) + [('clf', clf)])
else:
    pipeline = Pipeline([('pre', preprocessor), ('sel', selector), ('clf', clf)])

fitted = mlca_runtime.fit_record(pipeline, X, 'train', y=y)
mlca_runtime.save_model(fitted, 'model.pkl')

Xv = val[feature_cols]
yv = val[target]
proba = fitted.predict_proba(Xv)
classes = list(fitted.classes_)
if primary_metric == 'roc_auc' and len(classes) == 2:
    pos_label = classes[1]
    y_true = (yv == pos_label).astype(int).values
    mlca_runtime.record_metric('roc_auc', float(roc_auc_score(y_true, proba[:, 1])), 'val')

print(f"features={len(feature_cols)} train_rows={len(X)}")
print(f"classes={classes}")
