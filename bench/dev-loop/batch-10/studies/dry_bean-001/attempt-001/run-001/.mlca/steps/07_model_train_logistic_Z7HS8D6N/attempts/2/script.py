import mlca_runtime, json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())

targets = protocol['targets']
excluded = protocol['excluded_columns']
group = protocol.get('group_column')

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

drop = set(targets) | set(excluded)
if group:
    drop.add(group)
feat_cols = []
for c in train.columns:
    if c in drop:
        continue
    s = train[c]
    is_key = False
    if s.dtype == object or s.dtype.kind in 'iu':
        if s.nunique(dropna=False) == len(train):
            is_key = True
    if not is_key:
        feat_cols.append(c)

X_train = train[feat_cols]
y_train = train[targets[0]]

with open(inputs['05_preprocess_fit_preprocessing_G69EDXSB/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)
with open(inputs['06_features_select_features_32CBHHGS/selector.pkl'], 'rb') as f:
    sel = pickle.load(f)

raw = list(X_train.columns)

def in_names(o):
    return list(getattr(o, 'feature_names_in_', []))

sel_in = in_names(sel)
pre_in = in_names(pre)

steps = []
if sel_in == raw:
    steps.append(('selector', sel))
elif pre_in == raw:
    steps.append(('pre', pre))
    steps.append(('selector', sel))
else:
    raise ValueError(f'Feature schema mismatch: raw={raw[:3]} pre_in={pre_in[:3]} sel_in={sel_in[:3]}')

steps.append(('clf', LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000)))
model = Pipeline(steps)

mlca_runtime.fit_record(model, X_train, 'train', y=y_train)
mlca_runtime.save_model(model, 'model.pkl')

print(f'train rows={len(X_train)} feat_in={len(raw)} steps={[s[0] for s in model.steps]}')
print(f'target={targets[0]} n_classes={train[targets[0]].nunique()}')
