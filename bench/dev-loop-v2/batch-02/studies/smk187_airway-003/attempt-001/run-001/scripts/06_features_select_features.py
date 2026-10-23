import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
targets = protocol['targets']
excluded = protocol['excluded_columns']
group = protocol.get('group_column', None)
train = pd.read_parquet(inputs['train'])
with open(inputs['05_preprocess_fit_preprocessing_M7AXCKT1/preprocessor.pkl'], 'rb') as f:
    preprocessor = pickle.load(f)

if hasattr(preprocessor, 'feature_names_in_'):
    feature_cols = list(preprocessor.feature_names_in_)
else:
    feature_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group]

X_train = train[feature_cols].copy()
y_train = train[targets[0]]

n_out = preprocessor.transform(X_train.head(10)).shape[1]
k = min(20, n_out)
selector = SelectKBest(score_func=mutual_info_classif, k=k)
pipe = Pipeline([('preprocessor', preprocessor), ('selector', selector)])
fitted_pipe = mlca_runtime.fit_record(pipe, X_train, 'train', y_train)
mlca_runtime.save_model(fitted_pipe, 'selector.pkl')
print(f"Train shape: {X_train.shape}, selected k={k} of {n_out}")
