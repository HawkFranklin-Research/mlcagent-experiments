import json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import mlca_runtime

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths['_protocol'].read_text())

for key in ('targets', 'excluded_columns', 'group_column', 'task'):
    if key not in protocol:
        raise KeyError(f"Missing protocol key {key!r}; available top-level keys: {sorted(protocol)}")

targets = list(protocol['targets'])
excluded = set(protocol['excluded_columns'])
group = protocol['group_column']
task = protocol['task']
if len(targets) != 1:
    raise ValueError("Expected exactly one target")
target = targets[0]

train = pd.read_parquet(paths['train'])
y = train[target].copy()

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

prev = load_pkl(paths['07_model_train_logistic_GBE8KD6W/model.pkl'])
use_prev = (
    isinstance(prev, Pipeline)
    and hasattr(prev, 'feature_names_in_')
    and all(c in train.columns for c in prev.feature_names_in_)
)
pre = None

if use_prev:
    feature_cols = list(prev.feature_names_in_)
    steps = list(prev.steps)
else:
    pre = load_pkl(paths['05_preprocess_fit_preprocessing_03EVNKD3/preprocessor.pkl'])
    if hasattr(pre, 'feature_names_in_') and all(c in train.columns for c in pre.feature_names_in_):
        feature_cols = list(pre.feature_names_in_)
    else:
        n = len(train)
        base = [c for c in train.columns if c != target and c not in excluded and c != group]
        feature_cols = []
        for c in base:
            s = train[c]
            if s.dtype.kind in 'OUS' or s.dtype.kind in 'iu':
                if s.nunique(dropna=False) == n:
                    continue
            feature_cols.append(c)

bad = [c for c in feature_cols if c == target or c in excluded or c == group]
if bad:
    raise ValueError(f"Feature list contains forbidden columns: {bad[:5]}")

X = train[feature_cols].copy()

kw = {'random_state': 42}
if y.value_counts(normalize=True).min() < 0.1:
    kw['class_weight'] = 'balanced'
hgb = HistGradientBoostingClassifier(**kw)

if use_prev:
    steps[-1] = (steps[-1][0], hgb)
    pipe = Pipeline(steps)
else:
    sel = load_pkl(paths['06_features_select_features_9KY0CV9N/selector.pkl'])
    pipe = Pipeline([('preprocessor', pre), ('selector', sel), ('classifier', hgb)])

pipe = mlca_runtime.fit_record(pipe, X, 'train', y=y)
mlca_runtime.save_model(pipe, 'alternative_model.pkl')

if task.get('type') == 'binary' and task.get('primary_metric') == 'roc_auc':
    proba = pipe.predict_proba(X)
    cls = list(pipe.classes_)
    y_bin = (y.to_numpy() == cls[1]).astype(int)
    auc = roc_auc_score(y_bin, proba[:, 1])
    mlca_runtime.record_metric('train_roc_auc', float(auc), 'train')
    print(f"train roc_auc={auc:.4f}")

print(f"rows={len(train)} features={len(feature_cols)} saved alternative_model.pkl")
