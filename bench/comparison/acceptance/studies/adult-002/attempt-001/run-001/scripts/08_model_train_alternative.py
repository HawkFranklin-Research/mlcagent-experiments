import mlca_runtime
import json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

seed = 42
mlca_runtime.set_seed(seed)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
targets = protocol['targets']
excluded = protocol['excluded_columns']
group_col = protocol.get('group_column')
task = protocol['task']

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

with open(inputs['07_model_train_logistic_QH9VENHA/model.pkl'], 'rb') as f:
    base = pickle.load(f)

if isinstance(base, Pipeline) and base.steps:
    alt = Pipeline(base.steps[:-1] + [('classifier', HistGradientBoostingClassifier(random_state=seed))])
else:
    pre = pickle.load(open(inputs['05_preprocess_fit_preprocessing_5C9G845H/preprocessor.pkl'], 'rb'))
    sel = pickle.load(open(inputs['06_features_select_features_HP31G2ZF/selector.pkl'], 'rb'))
    alt = Pipeline([('preprocessor', pre), ('selector', sel),
                    ('classifier', HistGradientBoostingClassifier(random_state=seed))])

feature_cols = None
for _, obj in alt.steps:
    if hasattr(obj, 'feature_names_in_'):
        feature_cols = list(obj.feature_names_in_)
        break

if feature_cols is None:
    allowed = [c for c in train.columns
               if c not in targets + excluded and (group_col is None or c != group_col)]
    key_cols = []
    for c in allowed:
        s = train[c]
        if s.nunique() == len(train) and (s.dtype == 'object' or pd.api.types.is_string_dtype(s) or pd.api.types.is_integer_dtype(s)):
            key_cols.append(c)
    feature_cols = [c for c in allowed if c not in key_cols]

X_train = train[feature_cols]
y_train = train[targets[0]] if len(targets) == 1 else train[targets]

mlca_runtime.fit_record(alt, X_train, 'train', y=y_train)
mlca_runtime.save_model(alt, 'alternative_model.pkl')
