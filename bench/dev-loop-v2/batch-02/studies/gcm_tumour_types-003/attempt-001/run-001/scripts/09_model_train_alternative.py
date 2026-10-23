import mlca_runtime, json, pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
targets = protocol['targets']
excluded = protocol['excluded_columns']
group = protocol.get('group_column')
target = targets[0]

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])
feat_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group]
X_train, y_train = train[feat_cols], train[target]
X_val, y_val = val[feat_cols], val[target]

with open(inputs['latest_model'], 'rb') as f:
    saved = pickle.load(f)

if isinstance(saved, Pipeline):
    steps = list(saved.steps[:-1]) + [('clf', HistGradientBoostingClassifier(random_state=42))]
    pipe = Pipeline(steps)
else:
    pipe = Pipeline([('prep', saved), ('clf', HistGradientBoostingClassifier(random_state=42))])

model = mlca_runtime.fit_record(pipe, X_train, 'train', y_train)

proba = model.predict_proba(X_val)
classes = list(model.classes_)
present = [c for c in classes if (y_val == c).any()]
aucs = [roc_auc_score((y_val == c).astype(int), proba[:, classes.index(c)]) for c in present]
macro = float(np.mean(aucs)) if aucs else float('nan')
mlca_runtime.record_metric('roc_auc_ovr', macro, 'val')
print(f"fit rows={X_train.shape[0]} features={X_train.shape[1]} classes={len(classes)}")
print(f"val roc_auc_ovr={macro:.4f} (present={len(present)})")

mlca_runtime.save_model(model, 'alternative_model.pkl')
