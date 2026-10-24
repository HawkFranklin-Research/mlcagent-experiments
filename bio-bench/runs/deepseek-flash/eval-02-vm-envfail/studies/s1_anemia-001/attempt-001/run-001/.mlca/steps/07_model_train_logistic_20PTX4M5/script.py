import mlca_runtime
import json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def input_cols(obj):
    cols = getattr(obj, 'feature_names_in_', None)
    return list(cols) if cols is not None else None

def sel_has_pre(sel, pre, pre_cols):
    if not isinstance(sel, Pipeline) or pre_cols is None:
        return False
    for _, step in sel.named_steps.items():
        if step is pre:
            return True
        step_cols = input_cols(step)
        if step_cols is not None and step_cols == pre_cols:
            return True
    sel_cols = input_cols(sel)
    return sel_cols is not None and sel_cols == pre_cols and len(sel.named_steps) > 1

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
targets = protocol['targets']
excluded = protocol.get('excluded_columns', [])
group_col = protocol.get('group_column')
task = protocol['task']
if len(targets) != 1:
    raise ValueError('Expected exactly one target')
target = targets[0]

train = pd.read_parquet(inputs['train'])
with open(inputs['05_preprocess_fit_preprocessing_HCSJFKGG/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)
with open(inputs['06_features_select_features_5E6WDF2E/selector.pkl'], 'rb') as f:
    sel = pickle.load(f)

pre_cols = input_cols(pre)
if sel is not None and sel_has_pre(sel, pre, pre_cols):
    feature_pipeline = sel
else:
    steps = []
    if pre is not None:
        steps.append(('preprocessor', pre))
    if sel is not None:
        steps.append(('selector', sel))
    if not steps:
        raise ValueError('No preprocessing or selector available')
    feature_pipeline = Pipeline(steps) if len(steps) > 1 else steps[0][1]

first = feature_pipeline
cols = input_cols(first)
if cols is None:
    cols = input_cols(feature_pipeline)
if cols is None:
    cols = list(train.columns)

drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)

leak = [c for c in cols if c in drop]
if leak:
    raise ValueError(f'Feature list includes target/excluded/group columns: {leak}')
missing = [c for c in cols if c not in train.columns]
if missing:
    raise ValueError(f'Missing expected feature columns: {missing}')
feature_cols = cols

pipeline = Pipeline([
    ('features', feature_pipeline),
    ('model', LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000))
])
X_train = train[feature_cols]
y_train = train[target]
model = mlca_runtime.fit_record(pipeline, X_train, 'train', y=y_train)
mlca_runtime.save_model(model, 'model.pkl')

proba = model.predict_proba(X_train)
classes = list(model.classes_)
declared_pos = protocol.get('positive_label') or task.get('positive_label')
pos = declared_pos if declared_pos in classes else (classes[1] if len(classes) > 1 else classes[0])
j = classes.index(pos)
y_bin = (y_train == pos).astype(int)
auc = roc_auc_score(y_bin, proba[:, j])
mlca_runtime.record_metric('train_roc_auc', float(auc), 'train')
print('train rows', len(train), 'features', len(feature_cols), 'classes', classes, 'train roc_auc', round(float(auc), 4))
