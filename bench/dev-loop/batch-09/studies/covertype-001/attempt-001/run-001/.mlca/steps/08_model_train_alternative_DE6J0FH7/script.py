import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

SEED = 42
mlca_runtime.set_seed(SEED)

inp = mlca_runtime.inputs()
protocol = json.loads(inp['_protocol'].read_text())

for key in ['targets', 'excluded_columns', 'task']:
    if key not in protocol:
        raise KeyError(f"Missing _protocol key '{key}'. Available top-level keys: {list(protocol.keys())}")

targets = protocol['targets']
excluded = protocol['excluded_columns']
group_col = protocol.get('group_column')
task = protocol['task']
primary_metric = task['primary_metric']

train = pd.read_parquet(inp['train'])
val = pd.read_parquet(inp['val'])

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

pre = load_pkl(inp['05_preprocess_fit_preprocessing_0FJ6AMWD/preprocessor.pkl'])
sel = load_pkl(inp['06_features_select_features_79Q93RHQ/selector.pkl'])
model07 = load_pkl(inp['07_model_train_logistic_JDJV5YWV/model.pkl'])

if isinstance(model07, Pipeline):
    steps = list(model07.steps)
    steps[-1] = ('clf', HistGradientBoostingClassifier(random_state=SEED))
    alt = Pipeline(steps)
    feature_cols = list(model07.feature_names_in_) if hasattr(model07, 'feature_names_in_') else None
else:
    if isinstance(sel, Pipeline):
        base = sel
    else:
        base = Pipeline([('pre', pre), ('sel', sel)])
    alt = Pipeline([('base', base), ('clf', HistGradientBoostingClassifier(random_state=SEED))])
    feature_cols = list(base.feature_names_in_) if hasattr(base, 'feature_names_in_') else None

if feature_cols is None:
    def is_key_like(col, s):
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
            return s.nunique(dropna=True) == len(s)
        if pd.api.types.is_integer_dtype(s):
            name = str(col).lower()
            if any(tok in name for tok in ['id', 'key', 'index', 'code', 'row']):
                return s.nunique(dropna=True) == len(s)
        return False
    drop = set(targets) | set(excluded) | ({group_col} if group_col else set())
    feature_cols = [c for c in train.columns if c not in drop and not is_key_like(c, train[c])]

feature_cols = [c for c in feature_cols if c not in set(targets) | set(excluded) | ({group_col} if group_col else set())]
X_train = train[feature_cols].copy()
y_train = train[targets[0]]
alt = mlca_runtime.fit_record(alt, X_train, 'train', y=y_train)
mlca_runtime.save_model(alt, 'alternative_model.pkl')

if primary_metric == 'roc_auc_ovr' and hasattr(alt, 'predict_proba'):
    X_val = val[feature_cols].copy()
    y_val = val[targets[0]]
    proba = alt.predict_proba(X_val)
    clf = alt.steps[-1][1]
    labels = getattr(clf, 'classes_', None)
    if labels is not None:
        score = roc_auc_score(y_val, proba, multi_class='ovr', average='macro', labels=labels)
        mlca_runtime.record_metric(primary_metric, float(score), 'val')
        print(f"val {primary_metric}: {score:.4f}")

print(f"train rows: {len(train)}, features: {len(feature_cols)}")
