import mlca_runtime, json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
proto = json.loads(inp['_protocol'].read_text())
targets = proto['targets']
excluded = proto['excluded_columns']
group = proto.get('group_column')

train = pd.read_parquet(inp['train'])
val = pd.read_parquet(inp['val'])

with open(inp['06_preprocess_fit_preprocessing_FTEQSYVR/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)
with open(inp['07_features_retain_features_C5JWX6VS/selector_all.pkl'], 'rb') as f:
    sel = pickle.load(f)

pre_in = list(getattr(pre, 'feature_names_in_', []))
if not pre_in:
    drop = set(targets) | set(excluded) | ({group} if group else set())
    pre_in = [c for c in train.columns if c not in drop]

def wraps_pre(s):
    if not isinstance(s, Pipeline):
        return False
    names = list(getattr(list(s.named_steps.values())[0], 'feature_names_in_', []))
    return bool(names) and set(names) == set(pre_in)

frontend = [('selector', sel)] if wraps_pre(sel) else [('preprocessor', pre), ('selector', sel)]
pipe = Pipeline(frontend + [('clf', HistGradientBoostingClassifier(random_state=42))])

y = train[targets[0]].values
mlca_runtime.fit_record(pipe, train[pre_in], 'train', y=y)
mlca_runtime.save_model(pipe, 'alternative_model.pkl')

clf = pipe.named_steps['clf']
pos_label = clf.classes_[1]
y_val = val[targets[0]].values
proba = pipe.predict_proba(val[pre_in])[:, 1]
auc = roc_auc_score((y_val == pos_label).astype(int), proba)
mlca_runtime.record_metric('roc_auc', float(auc), 'val')
print(f"saved alternative_model.pkl; n_features={len(pre_in)}; val_roc_auc={auc:.4f}")
