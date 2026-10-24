import mlca_runtime, json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp['_protocol'].read_text())

targets = proto['targets']
excluded = set(proto['excluded_columns'])
group = proto.get('group_column')

train = pd.read_parquet(inp['train'])
val = pd.read_parquet(inp['val'])

with open(inp['05_preprocess_fit_preprocessing_B1ZJVKDM/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)
with open(inp['06_features_select_features_X6VS56A4/selector.pkl'], 'rb') as f:
    sel = pickle.load(f)

feats = list(getattr(pre, 'feature_names_in_', [])) or [
    c for c in train.columns
    if c not in targets and c not in excluded and c != group
]

Xtr = train[feats]
ytr = train[targets[0]]

pipe = Pipeline([
    ('pre', pre),
    ('sel', sel),
    ('clf', LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000)),
])
mlca_runtime.fit_record(pipe, Xtr, 'train', y=ytr)
mlca_runtime.save_model(pipe, 'model.pkl')

print(f"train rows={len(train)} features={len(feats)}")
print(f"pipeline steps={list(pipe.named_steps)}")
