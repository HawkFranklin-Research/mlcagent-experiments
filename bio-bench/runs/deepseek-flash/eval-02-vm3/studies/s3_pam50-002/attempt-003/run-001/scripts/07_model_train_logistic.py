import mlca_runtime as mlca
import json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

mlca.set_seed(42)
inputs = mlca.inputs()
proto = json.loads(inputs['_protocol'].read_text())
task = proto['task']
targets = proto['targets']
excluded = list(proto.get('excluded_columns', []))
group_col = proto.get('group_column')

train = pd.read_parquet(inputs['train'])
y = train[targets[0]]

with open(inputs['05_preprocess_fit_preprocessing_FS0JB8RV/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)
with open(inputs['06_features_select_features_5R6M6WZH/selector.pkl'], 'rb') as f:
    sel = pickle.load(f)

# Raw feature columns from the fitted preprocessor or selector
if hasattr(pre, 'feature_names_in_'):
    feats = list(pre.feature_names_in_)
elif hasattr(sel, 'feature_names_in_'):
    feats = list(sel.feature_names_in_)
else:
    drop = set(targets) | set(excluded)
    if group_col:
        drop.add(group_col)
    feats = [c for c in train.columns if c not in drop]

X = train[feats]

# Check whether the selector pipeline already contains the preprocessor
pre_feat = getattr(pre, 'feature_names_in_', None)
sel_includes_pre = False
if isinstance(sel, Pipeline):
    for _, step in sel.named_steps.items():
        if step is pre:
            sel_includes_pre = True
            break
        if pre_feat is not None and hasattr(step, 'feature_names_in_'):
            if type(step) is type(pre) and list(step.feature_names_in_) == list(pre_feat):
                sel_includes_pre = True
                break

clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000)
if sel_includes_pre:
    pipe = Pipeline(list(sel.steps) + [('clf', clf)])
else:
    pipe = Pipeline([('pre', pre), ('sel', sel), ('clf', clf)])

model = mlca.fit_record(pipe, X, 'train', y=y)
mlca.save_model(model, 'model.pkl')
print(f"Features: {len(feats)}, Train rows: {len(X)}")
print(f"Pipeline steps: {list(model.named_steps.keys())}")
