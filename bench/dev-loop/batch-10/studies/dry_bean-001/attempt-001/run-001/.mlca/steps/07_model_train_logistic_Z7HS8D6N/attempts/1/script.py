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
feat_cols = [c for c in train.columns if c not in drop]
for c in list(feat_cols):
    s = train[c]
    if (s.dtype == object) or (s.dtype.kind in 'iu'):
        if s.nunique(dropna=False) == len(train):
            feat_cols.remove(c)

X_train = train[feat_cols]
y_train = train[targets[0]]

with open(inputs['05_preprocess_fit_preprocessing_G69EDXSB/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)
with open(inputs['06_features_select_features_32CBHHGS/selector.pkl'], 'rb') as f:
    sel = pickle.load(f)

raw = list(X_train.columns)
def is_raw(o):
    return hasattr(o, 'feature_names_in_') and list(o.feature_names_in_) == raw

if is_raw(sel):
    steps = [('selector', sel)]
else:
    steps = [('pre', pre), ('selector', sel)]
steps.append(('clf', LogisticRegression(
    C=1.0, class_weight='balanced', max_iter=2000, solver='lbfgs',
    multi_class='auto' if hasattr(LogisticRegression(), 'multi_class') else None
)))
model = Pipeline(steps)

mlca_runtime.fit_record(model, X_train, 'train', y=y_train)
mlca_runtime.save_model(model, 'model.pkl')

print(f'train rows={len(X_train)} feat_in={len(raw)} steps={[s[0] for s in model.steps]}')
print(f'target={targets[0]} classes={sorted(train[targets[0]].unique().tolist())[:8]}')
