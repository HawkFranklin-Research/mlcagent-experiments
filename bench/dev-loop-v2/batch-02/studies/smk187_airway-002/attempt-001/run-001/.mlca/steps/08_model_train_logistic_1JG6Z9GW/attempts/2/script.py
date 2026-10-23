import mlca_runtime, json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
p = mlca_runtime.inputs()
proto = json.loads(p['_protocol'].read_text())
targets = proto['targets']
excluded = proto.get('excluded_columns') or []
group = proto.get('group_column')
train = pd.read_parquet(p['train'])
feats = [c for c in train.columns
         if c not in targets and c not in excluded and c != group]

with open(p['06_preprocess_fit_preprocessing_XWW5WTFN/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)
with open(p['07_features_select_features_3JX8FX9S/selector.pkl'], 'rb') as f:
    sel = pickle.load(f)

sel_in = getattr(sel, 'feature_names_in_', None)
sel_in = list(sel_in) if sel_in is not None else []
if sel_in and set(sel_in) == set(feats):
    steps = [('sel', sel)]
else:
    steps = [('pre', pre), ('sel', sel)]
steps.append(('clf', LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000)))
pipe = Pipeline(steps)

y = train[targets[0]]
mlca_runtime.fit_record(pipe, train[feats], 'train', y=y)
mlca_runtime.save_model(pipe, 'model.pkl')
print(f"n_train={len(train)} n_feats={len(feats)}")
