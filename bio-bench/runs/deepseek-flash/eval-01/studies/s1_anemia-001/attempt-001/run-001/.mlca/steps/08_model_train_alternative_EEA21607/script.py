import mlca_runtime, json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

inputs = mlca_runtime.inputs()
proto = json.loads(inputs['_protocol'].read_text())
targets = proto['targets']
excluded = proto.get('excluded_columns', [])
group_col = proto.get('group_column', None)
seed = 42
mlca_runtime.set_seed(seed)

train = pd.read_parquet(inputs['train'])
y = train[targets[0]]
min_frac = float(y.value_counts(normalize=True).min())
class_weight = 'balanced' if min_frac < 0.10 else None

with open(inputs['05_preprocess_fit_preprocessing_FMADE2YX/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)
with open(inputs['06_features_retain_features_SEM4Z3JD/selector_all.pkl'], 'rb') as f:
    sel = pickle.load(f)

pre_in = list(getattr(pre, 'feature_names_in_', []))
sel_in = list(getattr(sel, 'feature_names_in_', []))
# If the saved selector already expects the raw feature columns, it wraps preprocessing
# (side-by-side with pre): use it alone. Otherwise chain pre -> sel.
if pre_in and sel_in and set(pre_in) == set(sel_in):
    frontend = sel
else:
    frontend = Pipeline([('pre', pre), ('sel', sel)])

if pre_in:
    feat_cols = pre_in
else:
    feat_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group_col]

clf = HistGradientBoostingClassifier(random_state=seed, class_weight=class_weight)
pipe = Pipeline([('frontend', frontend), ('clf', clf)])
X_train = train[feat_cols]
mlca_runtime.fit_record(pipe, X_train, 'train', y)
mlca_runtime.save_model(pipe, 'alternative_model.pkl')
print(f"features: {len(feat_cols)}")
print(f"min class fraction: {min_frac:.3f}, balanced: {class_weight is not None}")
print("saved alternative_model.pkl")
