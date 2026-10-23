import mlca_runtime, json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp['_protocol'].read_text())
train = pd.read_parquet(inp['train'])
val = pd.read_parquet(inp['val'])
label = protocol['targets'][0]
excluded = list(protocol.get('excluded_columns') or [])
group = protocol.get('group_column')
drop = set(protocol['targets']) | set(excluded) | ({group} if group else set())
feature_cols = [c for c in train.columns if c not in drop]

def load(k):
    with open(inp[k], 'rb') as f:
        return pickle.load(f)

def fn(o):
    return list(getattr(o, 'feature_names_in_', []))

pre = load('06_preprocess_fit_preprocessing_XWW5WTFN/preprocessor.pkl')
sel = load('07_features_select_features_3JX8FX9S/selector.pkl')

raw = set(feature_cols)
clf = HistGradientBoostingClassifier(random_state=42)
if isinstance(sel, Pipeline) and fn(sel) and set(fn(sel)) == raw:
    steps = list(sel.steps) + [('clf', clf)]
else:
    steps = [('pre', pre), ('sel', sel), ('clf', clf)]
pipe = Pipeline(steps)

first = pipe.steps[0][1]
cols = list(getattr(first, 'feature_names_in_', feature_cols))
if not set(cols) <= set(train.columns):
    cols = feature_cols

Xtr = train[cols]
ytr = train[label]
pipe = mlca_runtime.fit_record(pipe, Xtr, 'train', ytr)

classes = pipe.named_steps['clf'].classes_
proba = pipe.predict_proba(val[cols])
if len(classes) == 2:
    yt = (val[label] == classes[1]).astype(int)
    auc = roc_auc_score(yt, proba[:, 1])
else:
    auc = roc_auc_score(val[label], proba, multi_class='ovr',
                        average='macro', labels=classes)
mlca_runtime.record_metric('roc_auc', float(auc), 'val')
mlca_runtime.save_model(pipe, 'alternative_model.pkl')
print(f"train_shape={Xtr.shape} n_classes={len(classes)} val_roc_auc={auc:.4f}")
