import mlca_runtime, json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
targets = protocol['targets']
excluded = protocol.get('excluded_columns', [])
group = protocol.get('group_column')

train = pd.read_parquet(inputs['train'])

with open(inputs['latest_model'], 'rb') as f:
    saved = pickle.load(f)

hgb = HistGradientBoostingClassifier(random_state=42)

if isinstance(saved, Pipeline) and len(saved.steps) >= 2:
    steps = list(saved.steps[:-1]) + [(saved.steps[-1][0], hgb)]
    pipe = Pipeline(steps)
    print("steps:", [s[0] for s in pipe.steps])
else:
    with open(inputs['06_features_select_features_RVKX3HX3/selector.pkl'], 'rb') as f:
        sel = pickle.load(f)
    pipe = Pipeline([('selector', sel), ('model', hgb)])
    print("fallback steps:", [s[0] for s in pipe.steps])

drop = list(targets) + list(excluded) + ([group] if group else [])
feature_cols = [c for c in train.columns if c not in drop]
X = train[feature_cols]
y = train[targets[0]]

pipe = mlca_runtime.fit_record(pipe, X, 'train', y)
mlca_runtime.save_model(pipe, 'alternative_model.pkl')
print("saved", X.shape, "classes:", y.nunique())
