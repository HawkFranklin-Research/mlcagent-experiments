import mlca_runtime
import json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()
protocol = json.loads(inp['_protocol'].read_text())
target = protocol['targets'][0]

train = pd.read_parquet(inp['train'])
val = pd.read_parquet(inp['val'])

with open(inp['05_preprocess_fit_preprocessing_CKA3B6GT/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)
with open(inp['06_features_select_features_XKN9FXQ4/selector.pkl'], 'rb') as f:
    sel = pickle.load(f)

pre_fn = getattr(pre, 'feature_names_in_', None)
sel_fn = getattr(sel, 'feature_names_in_', None)
raw_feats = [c for c in train.columns if c != target]

# Decide input columns for the pipeline frontend
if pre_fn is not None:
    input_cols = list(pre_fn)
elif sel_fn is not None and set(sel_fn).issubset(set(raw_feats)):
    input_cols = list(sel_fn)
else:
    input_cols = raw_feats

preple = getattr(pre, 'named_steps', None)
selsteps = getattr(sel, 'named_steps', None)

# Check whether sel already embeds preprocessing
sel_has_pre = False
if selsteps and preple:
    sel_has_pre = any(preple.get(k) is preple.get(k) and type(v).__name__ in (
        'ColumnTransformer', 'Pipeline', 'OneHotEncoder') for k, v in selsteps.items())

sel_builtin_pre = False
if selsteps:
    for _, v in selsteps.items():
        if type(v).__name__ in ('ColumnTransformer',):
            sel_builtin_pre = True
if type(sel).__name__ == 'ColumnTransformer':
    sel_builtin_pre = True

sel_expects_raw = (sel_fn is not None
                   and set(sel_fn).issubset(set(raw_feats))
                   and (sel_builtin_pre or sel_has_pre))

if sel_expects_raw:
    frontend_steps = [('sel', sel)]
else:
    frontend_steps = [('pre', pre), ('sel', sel)]

pipeline = Pipeline(
    frontend_steps + [
        ('clf', LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000))
    ]
)

X_train = train[input_cols]
y_train = train[target]
print("train shape:", X_train.shape, "pos rate:", float(y_train.astype(str).eq('yes').mean())
      if y_train.dtype == object else float(y_train.mean()))

fitted = mlca_runtime.fit_record(pipeline, X_train, 'train', y=y_train)
mlca_runtime.save_model(fitted, 'model.pkl')
print("saved model.pkl")
