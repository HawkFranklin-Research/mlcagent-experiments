import mlca_runtime, json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
targets = protocol['targets']
excluded = list(protocol.get('excluded_columns', []))
group = protocol.get('group_column')
train = pd.read_parquet(inputs['train'])

with open(inputs['06_preprocess_fit_preprocessing_6C1GR31N/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)
with open(inputs['07_features_select_features_2XVWGVRB/selector.pkl'], 'rb') as f:
    sel = pickle.load(f)

def first_input_cols(obj):
    if hasattr(obj, 'feature_names_in_'):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline):
        for s in obj.named_steps.values():
            if hasattr(s, 'feature_names_in_'):
                return list(s.feature_names_in_)
    return None

pre_cols = first_input_cols(pre)
sel_cols = first_input_cols(sel)

if pre_cols is not None and sel_cols is not None and set(pre_cols) == set(sel_cols):
    frontend = sel
    input_cols = list(sel_cols)
else:
    frontend = Pipeline([('pre', pre), ('sel', sel)])
    input_cols = list(pre_cols) if pre_cols is not None else list(sel_cols)

ban = set(targets) | set(excluded)
if group:
    ban.add(group)
input_cols = [c for c in input_cols if c not in ban]

missing = [c for c in input_cols if c not in train.columns]
if missing:
    raise ValueError(f'Missing input columns {missing[:5]}; train has {list(train.columns)[:5]}...')

model = Pipeline([
    ('fe', frontend),
    ('clf', LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000)),
])
X = train[input_cols]
y = train[targets[0]]
mlca_runtime.fit_record(model, X, 'train', y=y)
mlca_runtime.save_model(model, 'model.pkl')
print(f'train_shape={X.shape} classes={list(model.named_steps["clf"].classes_)}')
