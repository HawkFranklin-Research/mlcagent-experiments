import json, pickle
import pandas as pd
import mlca_runtime
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
proto = json.loads(inputs['_protocol'].read_text())
targets = proto['targets']
excluded = proto['excluded_columns']
group = proto.get('group_column')
task = proto['task']

train = pd.read_parquet(inputs['train'])
with open(inputs['06_preprocess_fit_preprocessing_MYPQXFW6/preprocessor.pkl'], 'rb') as fh:
    pre = pickle.load(fh)

try:
    raw_cols = list(pre.feature_names_in_)
except AttributeError:
    first = list(pre.named_steps.values())[0] if isinstance(pre, Pipeline) else pre
    raw_cols = list(first.feature_names_in_)

drop = set(targets) | set(excluded) | ({group} if group else set())
leak = drop & set(raw_cols)
if leak:
    raise ValueError(f"Preprocessor inputs include excluded columns: {sorted(leak)}")

target_name = targets[0]
X_train = train[raw_cols].copy()
y_train = train[target_name].values

probe = pre.transform(X_train.head(5))
n_out = int(probe.shape[1])
k = min(20, n_out)

pipe = Pipeline([
    ('pre', pre),
    ('sel', SelectKBest(mutual_info_classif, k=k)),
])
mlca_runtime.fit_record(pipe, X_train, 'train', y=y_train)
mlca_runtime.save_model(pipe, 'selector.pkl')
mlca_runtime.record_metric('n_features_in', float(n_out), 'train')
mlca_runtime.record_metric('k_selected', float(k), 'train')
print(f"train={X_train.shape} pre_out={n_out} k={k}")
