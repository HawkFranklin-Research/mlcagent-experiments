import mlca_runtime
import json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp['_protocol'].read_text())
target = protocol['targets'][0]
excluded = set(protocol.get('excluded_columns', [])) | {target}
grp = protocol.get('group_column')
if grp:
    excluded.add(grp)

train = pd.read_parquet(inp['train'])
with open(inp['05_preprocess_fit_preprocessing_CKA3B6GT/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)
with open(inp['06_features_select_features_XKN9FXQ4/selector.pkl'], 'rb') as f:
    sel = pickle.load(f)

pre_fn = getattr(pre, 'feature_names_in_', None)
sel_fn = getattr(sel, 'feature_names_in_', None)
raw_feats = [c for c in train.columns if c not in excluded]

if pre_fn is not None:
    input_cols = [c for c in pre_fn if c in train.columns]
elif sel_fn is not None and set(sel_fn).issubset(set(raw_feats)):
    input_cols = list(sel_fn)
else:
    input_cols = raw_feats

sel_expects_raw = (sel_fn is not None) and set(sel_fn).issubset(set(raw_feats))
frontend = [('sel', sel)] if sel_expects_raw else [('pre', pre), ('sel', sel)]

pipeline = Pipeline(frontend + [
    ('clf', LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000))
])

X_train = train[input_cols]
y_train = train[target]
if pd.api.types.is_numeric_dtype(y_train):
    pos_rate = float(y_train.mean())
else:
    pos_rate = float(y_train.eq('yes').mean())
print("train shape:", X_train.shape)
print("pos rate:", pos_rate)
print("frontend:", "sel-only" if sel_expects_raw else "pre+sel")

fitted = mlca_runtime.fit_record(pipeline, X_train, 'train', y=y_train)
mlca_runtime.save_model(fitted, 'model.pkl')
print("saved model.pkl")
