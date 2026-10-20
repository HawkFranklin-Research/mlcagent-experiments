import mlca_runtime, json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

proto = json.loads(inputs['_protocol'].read_text())
targets = proto['targets']
excluded = proto.get('excluded_columns', []) or []
group = proto.get('group_column')
target = targets[0]

train = pd.read_parquet(inputs['train'])

drop = set(targets) | set(excluded)
if group:
    drop.add(group)
base_cols = [c for c in train.columns if c not in drop]

key_cols = []
for c in base_cols:
    s = train[c]
    if s.dtype == object and s.nunique() == len(s) and s.nunique() > 5:
        key_cols.append(c)
feature_cols = [c for c in base_cols if c not in key_cols]

X_train = train[feature_cols]
y_train = train[target]

with open(inputs['05_preprocess_fit_preprocessing_XHNDRHCS/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)
with open(inputs['06_features_select_features_4TDFQMJC/selector.pkl'], 'rb') as f:
    sel = pickle.load(f)

lr = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000)

pre_out = None
try:
    pre_out = list(pre.get_feature_names_out())
except Exception:
    pass

sel_in = getattr(sel, 'feature_names_in_', None)
sel_in_list = list(sel_in) if sel_in is not None else None

if sel_in_list is not None and sel_in_list == feature_cols:
    model = Pipeline([('sel', sel), ('clf', lr)])
else:
    model = Pipeline([('pre', pre), ('sel', sel), ('clf', lr)])

mlca_runtime.fit_record(model, X_train, 'train', y_train)
mlca_runtime.save_model(model, 'model.pkl')

print(f"n_features={len(feature_cols)} n_train={len(X_train)}")
print(f"steps={list(model.named_steps)}")
