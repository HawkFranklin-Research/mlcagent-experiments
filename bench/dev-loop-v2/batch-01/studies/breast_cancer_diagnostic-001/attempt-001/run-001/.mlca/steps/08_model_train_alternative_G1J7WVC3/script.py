import mlca_runtime
import json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

seed = 42
mlca_runtime.set_seed(seed)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
train = pd.read_parquet(inputs['train'])

targets = list(protocol['targets'])
excluded = list(protocol.get('excluded_columns', []) or [])
group_col = protocol.get('group_column')

with open(inputs['07_model_train_logistic_2M248THN/model.pkl'], 'rb') as f:
    base = pickle.load(f)

if isinstance(base, Pipeline):
    pre_steps = list(base.named_steps.items())[:-1]
    input_cols = list(getattr(base, 'feature_names_in_', []))
else:
    pre_steps = []
    input_cols = list(getattr(base, 'feature_names_in_', []))

if not input_cols:
    with open(inputs['06_features_select_features_1B1C3SBH/selector.pkl'], 'rb') as f:
        sel = pickle.load(f)
    if isinstance(sel, Pipeline):
        pre_steps = list(sel.named_steps.items())
    else:
        pre_steps = [('selector', sel)]
    input_cols = list(getattr(sel, 'feature_names_in_', []))
    if not input_cols:
        input_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group_col]

X = train[input_cols]
y = train[targets[0]]

pipe = Pipeline(pre_steps + [('model', HistGradientBoostingClassifier(random_state=seed))])

mlca_runtime.fit_record(pipe, X, 'train', y=y)
mlca_runtime.save_model(pipe, 'alternative_model.pkl')

print(f"rows={len(train)} features={len(input_cols)} target={targets[0]}")
print(f"steps={[s[0] for s in pipe.named_steps]}")
