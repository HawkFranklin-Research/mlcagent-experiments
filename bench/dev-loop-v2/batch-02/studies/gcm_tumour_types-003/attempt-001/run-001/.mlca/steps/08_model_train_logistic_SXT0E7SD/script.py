import json, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import mlca_runtime

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()

protocol = json.loads(inp['_protocol'].read_text())
targets = protocol['targets']
excluded = protocol.get('excluded_columns', [])
group_col = protocol.get('group_column')

train = pd.read_parquet(inp['train'])

with open(inp['06_preprocess_fit_preprocessing_PDW4RHJ4/preprocessor.pkl'], 'rb') as f:
    preprocessor = pickle.load(f)
with open(inp['07_features_select_features_88EXJ3SM/selector.pkl'], 'rb') as f:
    selector = pickle.load(f)

raw_cols = list(getattr(preprocessor, 'feature_names_in_', []))
if not raw_cols:
    raw_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group_col]

lr = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000)
sel_inputs = list(getattr(selector, 'feature_names_in_', []))
if sel_inputs == raw_cols:
    full = Pipeline([('frontend', selector), ('model', lr)])
else:
    full = Pipeline([('preprocessing', preprocessor), ('selector', selector), ('model', lr)])

X = train[raw_cols]
y = train[targets[0]] if len(targets) == 1 else train[targets]

mlca_runtime.fit_record(full, X, 'train', y=y)
mlca_runtime.save_model(full, 'model.pkl')
print('features', len(raw_cols), 'train_rows', X.shape[0], 'classes', len(full.named_steps['model'].classes_))
