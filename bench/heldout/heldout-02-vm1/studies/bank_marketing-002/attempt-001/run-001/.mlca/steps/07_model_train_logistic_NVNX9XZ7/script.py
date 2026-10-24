import mlca_runtime, json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp['_protocol'].read_text())
train = pd.read_parquet(inp['train'])
targets = proto['targets']
excluded = proto['excluded_columns']
group = proto.get('group_column')
y = train[targets[0]]

with open(inp['05_preprocess_fit_preprocessing_VSWY5WH2/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)
with open(inp['06_features_select_features_4XA2PFRM/selector.pkl'], 'rb') as f:
    sel = pickle.load(f)

drop = set(targets) | set(excluded) | ({group} if group else set())
raw_cols = [c for c in train.columns if c not in drop]

def accepts_raw(obj):
    fin = getattr(obj, 'feature_names_in_', None)
    return fin is not None and list(fin) == raw_cols

clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000)
if accepts_raw(sel):
    model = Pipeline([('sel', sel), ('clf', clf)])
    input_cols = list(sel.feature_names_in_)
else:
    model = Pipeline([('prep', pre), ('sel', sel), ('clf', clf)])
    input_cols = list(pre.feature_names_in_) if hasattr(pre, 'feature_names_in_') else raw_cols

X = train[input_cols]
mlca_runtime.fit_record(model, X, 'train', y=y)
mlca_runtime.save_model(model, 'model.pkl')
print('features:', len(input_cols), 'rows:', len(X))
print('steps:', list(model.named_steps))
