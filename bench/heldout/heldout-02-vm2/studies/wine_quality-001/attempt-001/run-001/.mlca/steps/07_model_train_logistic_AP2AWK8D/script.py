import mlca_runtime, json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())

train = pd.read_parquet(inputs['train'])
targets = protocol['targets']
excluded = list(protocol['excluded_columns'])
group_col = protocol.get('group_column')

def is_key_like(s, n):
    if n <= 1:
        return False
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_integer_dtype(s):
        return s.nunique(dropna=True) == n
    return False

feature_cols = [
    c for c in train.columns
    if c not in targets and c not in excluded and c != group_col
    and not is_key_like(train[c], len(train))
]

X_train = train[feature_cols].copy()
y_train = train[targets[0]].to_numpy()

with open(inputs['05_preprocess_fit_preprocessing_CE7RFVQ7/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)
with open(inputs['06_features_select_features_JYFK2080/selector.pkl'], 'rb') as f:
    sel = pickle.load(f)

def accepts_raw(obj):
    fn = getattr(obj, 'feature_names_in_', None)
    return fn is not None and list(fn) == feature_cols

clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000)

if accepts_raw(sel):
    steps = list(sel.steps) if isinstance(sel, Pipeline) else [('frontend', sel)]
    pipe = Pipeline(steps + [('clf', clf)])
elif accepts_raw(pre):
    pipe = Pipeline([('pre', pre), ('sel', sel), ('clf', clf)])
else:
    raise RuntimeError(
        f"Cannot align frontend: pre.in={getattr(pre, 'feature_names_in_', None)}, "
        f"sel.in={getattr(sel, 'feature_names_in_', None)}"
    )

pipe = mlca_runtime.fit_record(pipe, X_train, 'train', y_train)
mlca_runtime.save_model(pipe, 'model.pkl')
print('features:', len(feature_cols), 'train rows:', len(X_train))
print('model.pkl saved:', type(pipe).__name__)
