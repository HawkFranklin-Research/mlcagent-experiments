import mlca_runtime
import json, pickle
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
paths = mlca_runtime.inputs()
protocol = json.loads(paths['_protocol'].read_text())
targets = protocol['targets']
excluded = protocol['excluded_columns']
group = protocol.get('group_column')
train = pd.read_parquet(paths['train'])

with open(paths['05_preprocess_fit_preprocessing_3K40DCMA/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)

target = targets[0]
n = len(train)
feature_cols = []
for c in train.columns:
    if c in targets or c in excluded or c == group:
        continue
    s = train[c]
    if s.dtype == object or str(s.dtype) == 'string':
        if s.nunique() == n:
            continue
    if pd.api.types.is_integer_dtype(s) and s.nunique() == n:
        continue
    feature_cols.append(c)

X_train = train[feature_cols]
y_train = train[target]
in_names = list(getattr(pre, 'feature_names_in_', feature_cols))
X_train_in = X_train[in_names]

n_features = len(pre.get_feature_names_out())
k = min(20, n_features)

final = Pipeline([('preprocessor', pre), ('selector', SelectKBest(mutual_info_classif, k=k))])
mlca_runtime.fit_record(final, X_train_in, 'train', y=y_train)
mlca_runtime.save_model(final, 'selector.pkl')
print(f"k={k}, features_in={n_features}, X={X_train_in.shape}")
