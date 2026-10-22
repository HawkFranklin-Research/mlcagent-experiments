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
min_frac = y.value_counts(normalize=True).min()
class_weight = 'balanced' if min_frac < 0.10 else None

with open(inputs['05_preprocess_fit_preprocessing_FMADE2YX/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)
with open(inputs['06_features_retain_features_SEM4Z3JD/selector_all.pkl'], 'rb') as f:
    sel = pickle.load(f)

clf = HistGradientBoostingClassifier(random_state=seed, class_weight=class_weight)
pipe = Pipeline([('pre', pre), ('sel', sel), ('clf', clf)])

feat_cols = list(pre.feature_names_in_) if hasattr(pre, 'feature_names_in_') else [
    c for c in train.columns if c not in targets and c not in excluded and c != group_col
]
X_train = train[feat_cols]
mlca_runtime.fit_record(pipe, X_train, 'train', y)
mlca_runtime.save_model(pipe, 'alternative_model.pkl')
print(f"features: {len(feat_cols)}")
print(f"min class fraction: {min_frac:.3f}, balanced: {class_weight is not None}")
print("saved alternative_model.pkl")
