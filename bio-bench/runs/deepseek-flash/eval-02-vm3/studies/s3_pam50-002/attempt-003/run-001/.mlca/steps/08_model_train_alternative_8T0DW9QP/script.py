import mlca_runtime, json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
targets = protocol['targets']
excluded = list(protocol['excluded_columns'])
group_col = protocol.get('group_column')
task = protocol['task']

train = pd.read_parquet(inputs['train'])

with open(inputs['05_preprocess_fit_preprocessing_FS0JB8RV/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)
with open(inputs['06_features_select_features_5R6M6WZH/selector.pkl'], 'rb') as f:
    sel = pickle.load(f)

pre_in = getattr(pre, 'feature_names_in_', None)
sel_in = getattr(sel, 'feature_names_in_', None)
pre_in = list(pre_in) if pre_in is not None else None
sel_in = list(sel_in) if sel_in is not None else None

def is_keylike(s):
    if s.dtype.kind in 'iu' and s.nunique() == len(s):
        return True
    if s.dtype == object and s.nunique() == len(s):
        return True
    return False

if sel_in is not None and pre_in is not None and sel_in == pre_in:
    raw_cols = sel_in
    steps = [('selector', sel)]
elif pre_in is not None:
    raw_cols = pre_in
    steps = [('preprocessor', pre), ('selector', sel)]
else:
    raw_cols = [c for c in train.columns
                if c not in targets and c not in excluded and c != group_col]
    raw_cols = [c for c in raw_cols if not is_keylike(train[c])]
    steps = [('preprocessor', pre), ('selector', sel)]

clf = HistGradientBoostingClassifier(random_state=42)
pipe = Pipeline(steps + [('model', clf)])

X_train = train[raw_cols]
y_train = train[targets[0]]

fitted = mlca_runtime.fit_record(pipe, X_train, 'train', y_train)
mlca_runtime.save_model(fitted, 'alternative_model.pkl')

print(f"features={len(raw_cols)} classes={int(y_train.nunique())} rows={len(train)}")
