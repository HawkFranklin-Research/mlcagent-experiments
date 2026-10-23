import json, pickle
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
target = protocol['targets'][0]

train = pd.read_parquet(inputs['train'])
with open(inputs['latest_model'], 'rb') as f:
    model = pickle.load(f)

if hasattr(model, 'feature_names_in_'):
    feature_cols = list(model.feature_names_in_)
elif isinstance(model, Pipeline):
    feature_cols = list(model.steps[0][1].feature_names_in_)
else:
    raise RuntimeError('Cannot determine feature columns from saved model.')

X = train[feature_cols]
y = train[target]

min_prop = y.value_counts(normalize=True).min()
cw = 'balanced' if min_prop < 0.10 else None
new_est = HistGradientBoostingClassifier(random_state=42, class_weight=cw)

if isinstance(model, Pipeline):
    new_pipe = clone(model)
    new_pipe.steps[-1] = ('clf', new_est)
else:
    new_pipe = Pipeline([('clf', new_est)])

mlca_runtime.fit_record(new_pipe, X, 'train', y=y)
mlca_runtime.save_model(new_pipe, 'alternative_model.pkl')
print(f"train={train.shape} feats={len(feature_cols)} min_prop={min_prop:.3f} done")
