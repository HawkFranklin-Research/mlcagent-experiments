import json, pickle
import mlca_runtime
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
targets = list(protocol['targets'])
excluded = list(protocol.get('excluded_columns') or [])
group = protocol.get('group_column')

train = pd.read_parquet(inputs['train'])

with open(inputs['05_preprocess_fit_preprocessing_DXVQHGBE/preprocessor.pkl'], 'rb') as f:
    pre = pickle.load(f)

n_rows = len(train)
reserved = set(targets) | set(excluded) | ({group} if group else set())
cand = [c for c in train.columns if c not in reserved]
drop_ids = []
for c in cand:
    s = train[c]
    if s.nunique(dropna=False) != n_rows:
        continue
    if pd.api.types.is_bool_dtype(s) or pd.api.types.is_float_dtype(s):
        continue
    drop_ids.append(c)
feat_cols = [c for c in cand if c not in drop_ids]

raw_cols = list(pre.feature_names_in_) if hasattr(pre, 'feature_names_in_') else feat_cols
X_train = train[raw_cols]
y_train = train[targets[0]] if len(targets) == 1 else train[targets]

X_probe = pre.transform(X_train)
n_out = X_probe.shape[1]
k = min(20, n_out)

selector = SelectKBest(mutual_info_classif, k=k)
pipeline = Pipeline([('pre', pre), ('selector', selector)])
mlca_runtime.fit_record(pipeline, X_train, 'train', y=y_train)

mlca_runtime.save_model(pipeline, 'selector.pkl')
mlca_runtime.record_metric('n_features_after_preprocessing', float(n_out), 'train')
mlca_runtime.record_metric('n_features_selected', float(k), 'train')

print(f"train rows={n_rows}, raw features={len(raw_cols)}")
print(f"after preprocessing={n_out}, selected k={k}")
