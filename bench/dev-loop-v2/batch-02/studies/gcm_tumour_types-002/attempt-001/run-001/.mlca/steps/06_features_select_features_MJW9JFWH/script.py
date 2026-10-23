import mlca_runtime
import json, pickle
import pandas as pd
from functools import partial
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif

inputs = mlca_runtime.inputs()
train = pd.read_parquet(inputs['train'])
protocol = json.loads(inputs['_protocol'].read_text())

targets = protocol['targets']
excluded = protocol['excluded_columns']
group = protocol.get('group_column')
target = targets[0]

feature_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group]
# drop key-like text codes (all distinct object columns)
feature_cols = [c for c in feature_cols if not (train[c].dtype == object and train[c].nunique(dropna=False) == len(train))]

with open(inputs['05_preprocess_fit_preprocessing_0GNR753P/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)

if hasattr(pre, 'feature_names_in_'):
    feature_cols = list(pre.feature_names_in_)

if target in feature_cols:
    raise ValueError("Target leaked into feature columns")
if any(c in feature_cols for c in excluded):
    raise ValueError("Excluded column leaked into feature columns")

X = train[feature_cols]
y = train[target]

probe = X.head(50)
n_out = pre.transform(probe).shape[1] if hasattr(pre, 'transform') else len(feature_cols)
k = min(20, n_out)

selector = SelectKBest(score_func=partial(mutual_info_classif, random_state=42), k=k)
pipe = Pipeline([('preprocessor', pre), ('selector', selector)])

mlca_runtime.set_seed(42)
mlca_runtime.fit_record(pipe, X, 'train', y=y)
mlca_runtime.save_model(pipe, 'selector.pkl')
print(f"features={len(feature_cols)} pre_out={n_out} k={k}")
