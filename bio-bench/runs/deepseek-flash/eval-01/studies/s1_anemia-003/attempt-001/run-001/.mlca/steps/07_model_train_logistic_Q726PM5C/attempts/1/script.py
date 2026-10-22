import mlca_runtime
import pandas as pd, pickle, json
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from pathlib import Path

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(Path(inputs['_protocol']).read_text())
targets = protocol['targets']
excluded = protocol['excluded_columns']
group_col = protocol.get('group_column')
task = protocol['task']

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

with open(inputs['05_preprocess_fit_preprocessing_2E873D38/preprocessor.pkl'], 'rb') as f:
    preprocessor = pickle.load(f)
with open(inputs['06_features_select_features_CMGHE25G/selector.pkl'], 'rb') as f:
    selector = pickle.load(f)

drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)
feature_cols = [c for c in train.columns if c not in drop]
target = targets[0]
X = train[feature_cols]
y = train[target]

pre_raw = list(getattr(preprocessor, 'feature_names_in_', []) or [])
sel_raw = list(getattr(selector, 'feature_names_in_', []) or [])
clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000)

# RL-0011/0027: if selector is a Pipeline already fitted on the same raw columns,
# it wraps preprocessing; otherwise chain preprocessor -> selector -> clf.
if isinstance(selector, Pipeline) and pre_raw and sel_raw == pre_raw:
    pipeline = Pipeline(list(selector.steps) + [('clf', clf)])
else:
    pipeline = Pipeline([('pre', preprocessor), ('sel', selector), ('clf', clf)])

fitted = mlca_runtime.fit_record(pipeline, X, 'train', y=y)
mlca_runtime.save_model(fitted, 'model.pkl')

print(f"features={len(feature_cols)} rows={len(X)}")
print(f"classes={list(fitted.classes_)}")
