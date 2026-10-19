import mlca_runtime, json, pickle
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())

targets = protocol['targets']
excluded = protocol.get('excluded_columns', [])
group_col = protocol.get('group_column')
task = protocol['task']
primary_metric = task['primary_metric']

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

with open(inputs['05_preprocess_fit_preprocessing_G69EDXSB/preprocessor.pkl'], 'rb') as f:
    preprocessor = pickle.load(f)
with open(inputs['06_features_select_features_32CBHHGS/selector.pkl'], 'rb') as f:
    selector = pickle.load(f)

# Build the base pipeline, avoiding double-chaining of preprocessing
if isinstance(selector, Pipeline):
    first_name, first_step = selector.steps[0]
    includes_pre = (first_name in ('preprocessor', 'preprocess', 'preprocessing') or
                    hasattr(first_step, 'transformers') or
                    first_step.__class__.__name__ == 'ColumnTransformer')
    base_steps = list(selector.steps) if includes_pre else [('preprocessor', preprocessor), ('selector', selector)]
else:
    base_steps = [('preprocessor', preprocessor), ('selector', selector)]

clf = HistGradientBoostingClassifier(random_state=42)
pipeline = Pipeline(base_steps + [('clf', clf)])

# Determine raw feature columns from the first fitted step
first = pipeline.steps[0][1]
if hasattr(first, 'feature_names_in_'):
    feature_cols = list(first.feature_names_in_)
else:
    cand = [c for c in train.columns if c not in targets and c not in excluded and c != group_col]
    keep = []
    for c in cand:
        s = train[c]
        if s.dtype == object or pd.api.types.is_string_dtype(s):
            if s.nunique() == len(s):
                continue
        elif pd.api.types.is_integer_dtype(s) and s.nunique() == len(s):
            if s.is_monotonic_increasing or (s.max() - s.min() == len(s) - 1):
                continue
        keep.append(c)
    feature_cols = keep

X_train = train[feature_cols]
y_train = train[targets[0]]
X_val = val[feature_cols]
y_val = val[targets[0]]

model = mlca_runtime.fit_record(pipeline, X_train, 'train', y=y_train)

if primary_metric == 'roc_auc_ovr':
    proba = model.predict_proba(X_val)
    classes = model.classes_
    present = np.isin(classes, y_val.unique())
    if present.all():
        auc = roc_auc_score(y_val, proba, multi_class='ovr', average='macro', labels=classes)
    else:
        aucs = [roc_auc_score((y_val == c).astype(int), proba[:, j])
                for j, c in enumerate(classes) if c in y_val.unique()]
        auc = np.mean(aucs)
        print(f"Missing classes in val: {set(classes) - set(y_val.unique())}")
    mlca_runtime.record_metric('roc_auc_ovr', float(auc), 'val')

mlca_runtime.save_model(model, 'alternative_model.pkl')
print(f"Saved alternative_model.pkl; train {X_train.shape}, val {X_val.shape}")
