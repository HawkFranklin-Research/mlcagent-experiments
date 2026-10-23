import mlca_runtime, json, pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
train = pd.read_parquet(inp['train'])
proto = json.loads(inp['_protocol'].read_text())
targets = proto['targets']
excluded = proto['excluded_columns']
group = proto.get('group_column')

with open(inp['05_preprocess_fit_preprocessing_4ZKXP498/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)
with open(inp['06_features_select_features_E9CTG9EJ/selector.pkl'], 'rb') as f:
    sel = pickle.load(f)

feat_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group]
sel_in = getattr(sel, 'feature_names_in_', None)
clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000)

if sel_in is not None and set(sel_in) == set(feat_cols):
    cols = list(sel_in)
    pipe = Pipeline([('sel', sel), ('clf', clf)])
else:
    cols = feat_cols
    pipe = Pipeline([('pre', pre), ('sel', sel), ('clf', clf)])

X = train[cols]
y = train[targets[0]]
pipe = mlca_runtime.fit_record(pipe, X, 'train', y)
mlca_runtime.save_model(pipe, 'model.pkl')
print('features', X.shape[1], 'train rows', X.shape[0], 'fitted')
