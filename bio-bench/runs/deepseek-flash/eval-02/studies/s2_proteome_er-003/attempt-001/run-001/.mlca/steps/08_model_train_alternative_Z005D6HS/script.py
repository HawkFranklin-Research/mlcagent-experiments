import json, pickle
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

SEED = 42
mlca_runtime.set_seed(SEED)
inp = mlca_runtime.inputs()
proto = json.loads(inp['_protocol'].read_text())
targets = list(proto['targets'])
task = proto['task']
primary = task['primary_metric']
ycol = targets[0]

train = pd.read_parquet(inp['train'])
val = pd.read_parquet(inp['val'])

with open(inp['07_model_train_logistic_8VZRZDX0/model.pkl'], 'rb') as f:
    base = pickle.load(f)
if not isinstance(base, Pipeline):
    raise RuntimeError('expected a saved sklearn Pipeline, got %r' % type(base))
feat_cols = list(base.feature_names_in_)
steps = list(base.steps)
steps[-1] = ('model', HistGradientBoostingClassifier(random_state=SEED))
pipe = Pipeline(steps)

X = train[feat_cols]
y = train[ycol]
pipe = mlca_runtime.fit_record(pipe, X, 'train', y=y)
mlca_runtime.save_model(pipe, 'alternative_model.pkl')
print('saved alternative_model.pkl', X.shape, 'n_feat', len(feat_cols))

if primary == 'roc_auc' and hasattr(pipe, 'predict_proba'):
    classes = list(pipe.classes_)
    if len(classes) == 2:
        pos = classes[1]
        Xv = val[feat_cols]
        p = pipe.predict_proba(Xv)
        j = classes.index(pos)
        try:
            auc = roc_auc_score((val[ycol].values == pos).astype(int), p[:, j])
            mlca_runtime.record_metric('val_roc_auc', float(auc), 'val')
            print('val_roc_auc', round(float(auc), 4), 'pos_label', pos)
        except Exception as exc:
            print('skip val auc:', type(exc).__name__)
