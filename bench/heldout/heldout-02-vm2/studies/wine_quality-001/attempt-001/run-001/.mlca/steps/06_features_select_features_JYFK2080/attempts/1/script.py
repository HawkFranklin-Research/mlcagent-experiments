import mlca_runtime, json, pickle
from functools import partial
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs['_protocol'].read_text())
targets = list(protocol['targets'])
excluded = list(protocol['excluded_columns'])
group_col = protocol.get('group_column')

train = pd.read_parquet(inputs['train'])

with open(inputs['05_preprocess_fit_preprocessing_CE7RFVQ7/preprocessor.pkl'], 'rb') as f:
    preproc = pickle.load(f)

if hasattr(preproc, 'feature_names_in_') and preproc.feature_names_in_ is not None:
    feats = [c for c in list(preproc.feature_names_in_) if c in train.columns]
else:
    feats = [c for c in train.columns if c not in targets + excluded and c != group_col]

already_has_sel = isinstance(preproc, Pipeline) and any(
    isinstance(s, SelectKBest) or 'select' in n.lower() for n, s in preproc.named_steps.items()
)

if already_has_sel:
    pipe = preproc
else:
    Xt = preproc.transform(train[feats])
    n_out = int(Xt.shape[1])
    k = int(min(20, n_out))
    score_fn = partial(mutual_info_classif, random_state=42)
    sel = SelectKBest(score_fn, k=k)
    if isinstance(preproc, Pipeline):
        pipe = Pipeline(list(preproc.steps) + [('selectkbest', sel)])
    else:
        pipe = Pipeline([('preprocess', preproc), ('selectkbest', sel)])

X = train[feats]
y = train[targets[0]]
mlca_runtime.fit_record(pipe, X, 'train', y=y)

mlca_runtime.save_model(pipe, 'selector.pkl')

n_sel = None
if isinstance(pipe, Pipeline):
    for step in pipe.named_steps.values():
        if isinstance(step, SelectKBest):
            n_sel = int(step.get_support().sum())
            break
if n_sel is not None:
    mlca_runtime.record_metric('n_features_selected', float(n_sel), 'train')

print('raw_feature_cols', len(feats))
print('selected', n_sel)
print('saved selector.pkl')
