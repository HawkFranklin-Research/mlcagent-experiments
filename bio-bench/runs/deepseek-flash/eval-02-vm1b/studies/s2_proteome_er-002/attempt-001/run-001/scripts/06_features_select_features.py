import mlca_runtime, pickle, json
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())

train = pd.read_parquet(inputs['train'])
with open(inputs['05_preprocess_fit_preprocessing_BE6QCTJ4/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)

targets = protocol['targets']
y = train[targets[0]]

if hasattr(pre, 'feature_names_in_'):
    feats = list(pre.feature_names_in_)
else:
    excluded = set(protocol['excluded_columns'] if isinstance(protocol['excluded_columns'], list)
                   else protocol['excluded_columns'].get('examples', []))
    group = protocol.get('group_column')
    feats = [c for c in train.columns
             if c not in targets and c not in excluded and c != group]

X = train[feats]
Xt = pre.transform(X)
k = max(1, min(20, Xt.shape[1]))

selector_pipeline = Pipeline([
    ('pre', clone(pre)),
    ('select', SelectKBest(score_func=mutual_info_classif, k=k)),
])

fitted = mlca_runtime.fit_record(selector_pipeline, X, 'train', y)
mlca_runtime.save_model(fitted, 'selector.pkl')
mlca_runtime.record_metric('n_input_features', float(len(feats)), 'train')
mlca_runtime.record_metric('n_selected_features', float(k), 'train')
print(f"input features {len(feats)}, transformed width {Xt.shape[1]}, k={k}")
