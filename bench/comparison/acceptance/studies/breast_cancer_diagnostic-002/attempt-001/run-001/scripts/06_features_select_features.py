import json, pickle
import pandas as pd
import mlca_runtime
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())

missing = [k for k in ('targets', 'excluded_columns', 'task') if k not in protocol]
if missing:
    raise KeyError(
        f"Protocol missing keys {missing}; available top-level keys: {list(protocol.keys())}"
    )

targets = list(protocol['targets'])
excluded = list(protocol['excluded_columns'])
group = protocol.get('group_column')
target = targets[0]
_ = protocol['task']['primary_metric']

train = pd.read_parquet(inputs['train'])

with open(inputs['05_preprocess_fit_preprocessing_VDSDFD6K/preprocessor.pkl'], 'rb') as f:
    preprocessor = pickle.load(f)

skip = set(targets) | set(excluded) | ({group} if group else set())

def is_keylike(s):
    nm = str(s).lower()
    return nm in {'id', 'index', 'row_id', 'key', 'rowid'} or nm.endswith('_id') or nm.endswith('index')

if hasattr(preprocessor, 'feature_names_in_'):
    feature_cols = [c for c in list(preprocessor.feature_names_in_)
                    if c in train.columns and c not in skip and not is_keylike(c)]
else:
    feature_cols = [c for c in train.columns if c not in skip and not is_keylike(c)]

X_train = train[feature_cols]
y_train = train[target]

try:
    k = min(20, preprocessor.transform(X_train.iloc[:5]).shape[1])
except Exception:
    k = min(20, X_train.shape[1])

if hasattr(preprocessor, 'set_output'):
    try:
        preprocessor.set_output(transform='pandas')
    except Exception:
        pass

has_sel = isinstance(preprocessor, Pipeline) and any(
    ('select' in n.lower() or 'kbest' in n.lower()) for n in preprocessor.named_steps
)

if has_sel:
    pipe = preprocessor
else:
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    pipe = Pipeline([('preprocess', preprocessor), ('select', selector)])

mlca_runtime.fit_record(pipe, X_train, 'train', y=y_train)
mlca_runtime.save_model(pipe, 'selector.pkl')

print(f"features={X_train.shape[1]} k={k} rows={X_train.shape[0]}")
