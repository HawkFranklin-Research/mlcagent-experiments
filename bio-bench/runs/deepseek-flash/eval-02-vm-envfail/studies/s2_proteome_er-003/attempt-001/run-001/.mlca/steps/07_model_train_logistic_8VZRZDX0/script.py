import mlca_runtime, json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp['_protocol'].read_text())

targets = list(proto['targets'])
target = targets[0]
excluded = set(proto['excluded_columns'])
group = proto.get('group_column')

train = pd.read_parquet(inp['train'])

with open(inp['05_preprocess_fit_preprocessing_B1ZJVKDM/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)
with open(inp['06_features_select_features_X6VS56A4/selector.pkl'], 'rb') as f:
    sel = pickle.load(f)


def fni(obj):
    v = getattr(obj, 'feature_names_in_', None)
    return list(v) if v is not None else None


pre_f = fni(pre)
raw = pre_f if pre_f is not None else [
    c for c in train.columns if c not in set(targets) | excluded and c != group
]

sel_f = fni(sel)
if sel_f is None and isinstance(sel, Pipeline):
    sel_f = fni(list(sel.named_steps.values())[0])

sel_uses_raw = sel_f is not None and set(sel_f) == set(raw)

clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000)
if sel_uses_raw:
    feats = [c for c in sel_f]
    pipe = Pipeline([('sel', sel), ('clf', clf)])
else:
    feats = list(raw)
    pipe = Pipeline([('pre', pre), ('sel', sel), ('clf', clf)])

Xtr = train[feats]
ytr = train[target]
mlca_runtime.fit_record(pipe, Xtr, 'train', y=ytr)
mlca_runtime.save_model(pipe, 'model.pkl')

print(f"train rows={len(train)} features={len(feats)}")
print(f"selector_included_preprocessing={sel_uses_raw}")
print(f"pipeline steps={list(pipe.named_steps)}")
