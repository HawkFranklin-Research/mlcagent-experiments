import json, pickle
import mlca_runtime
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

protocol = json.loads(inputs['_protocol'].read_text())
targets = protocol['targets']
excluded = list(protocol['excluded_columns'])
group_col = protocol.get('group_column')
target = targets[0]

train = pd.read_parquet(inputs['train'])

with open(inputs['05_preprocess_fit_preprocessing_F9B0NJ26/preprocessor.pkl'], 'rb') as f:
    prep = pickle.load(f)
with open(inputs['06_features_select_features_AA6KHSXZ/selector.pkl'], 'rb') as f:
    sel = pickle.load(f)

def fnames(obj):
    if obj is None:
        return None
    try:
        if hasattr(obj, 'feature_names_in_'):
            return list(obj.feature_names_in_)
    except Exception:
        pass
    return None

def has_ct(obj):
    if isinstance(obj, ColumnTransformer):
        return True
    if isinstance(obj, Pipeline):
        return any(isinstance(s, ColumnTransformer) for s in obj.named_steps.values())
    return False

prep_in = fnames(prep)
sel_in = fnames(sel)

if prep is None:
    frontend = sel
    fit_cols = sel_in
elif sel is None:
    frontend = prep
    fit_cols = prep_in
elif sel_in is not None and prep_in is not None and len(sel_in) == len(prep_in) and set(sel_in) == set(prep_in):
    frontend = sel
    fit_cols = sel_in
else:
    frontend = Pipeline([('prep', prep), ('sel', sel)])
    fit_cols = prep_in if prep_in else sel_in

if fit_cols is None:
    raise RuntimeError("Cannot determine feature columns from saved objects")

drop = set(targets) | set(excluded)
if group_col:
    drop.add(group_col)
fit_cols = [c for c in fit_cols if c not in drop]

mask = train[target].notna()
train_m = train.loc[mask]
y_raw = train_m[target]
y_str = y_raw.astype(str).str.strip().str.rstrip('.')
uniq = sorted(y_str.dropna().unique())
pos_label = uniq[-1]
y = (y_str == pos_label).astype(int)
X_train = train_m[fit_cols]

model = Pipeline([
    ('frontend', frontend),
    ('clf', LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000, random_state=42)),
])

mlca_runtime.fit_record(model, X_train, 'train', y=y)
mlca_runtime.save_model(model, 'model.pkl')

print(f"train rows: {len(X_train)}, features: {len(fit_cols)}")
print(f"positive label: {pos_label}, positive rate: {y.mean():.4f}")
