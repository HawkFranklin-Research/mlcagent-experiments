import mlca_runtime, json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs['_protocol'].read_text())
targets = proto['targets']
excluded = proto.get('excluded_columns', []) or []
group = proto.get('group_column')

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

with open(inputs['05_preprocess_fit_preprocessing_CQ5BZKTY/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)
with open(inputs['06_features_select_features_XJNGW7H2/selector.pkl'], 'rb') as f:
    sel = pickle.load(f)

drop = set(targets) | set(excluded)
if group:
    drop.add(group)
feat_cols = []
for c in train.columns:
    if c in drop:
        continue
    s = train[c]
    if s.dtype == object and s.nunique() == len(s):
        continue
    if s.dtype.kind in 'iu' and s.nunique() == len(s):
        continue
    feat_cols.append(c)

clf = HistGradientBoostingClassifier(random_state=42)
if isinstance(sel, Pipeline) and len(sel.named_steps) > 1:
    model = Pipeline([('sel', sel), ('clf', clf)])
else:
    model = Pipeline([('pre', pre), ('sel', sel), ('clf', clf)])

X = train[feat_cols]
y = train[targets[0]]
mlca_runtime.fit_record(model, X, 'train', y)

Xv = val[feat_cols]
yv = val[targets[0]]
proba = model.predict_proba(Xv)
auc = roc_auc_score(yv, proba, multi_class='ovr', average='macro', labels=model.classes_)
mlca_runtime.record_metric('roc_auc_ovr', float(auc), 'val')

mlca_runtime.save_model(model, 'alternative_model.pkl')
print(f"features={len(feat_cols)} train_rows={len(train)} val_rows={len(val)}")
print(f"val_roc_auc_ovr={auc:.4f}")
