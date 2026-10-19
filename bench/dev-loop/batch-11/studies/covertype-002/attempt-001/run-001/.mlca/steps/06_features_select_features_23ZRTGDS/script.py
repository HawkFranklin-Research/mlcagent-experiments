import json, pickle
from functools import partial
import pandas as pd
import mlca_runtime
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
targets = list(protocol['targets'])
excluded = list(protocol.get('excluded_columns', []))
group_col = protocol.get('group_column')

train_df = pd.read_parquet(inputs['train'])
with open(inputs['05_preprocess_fit_preprocessing_8APC2384/preprocessor.pkl'], 'rb') as f:
    preprocessor = pickle.load(f)

drop = set(targets) | set(excluded) | ({group_col} if group_col else set())
if hasattr(preprocessor, 'feature_names_in_'):
    prep_cols = list(preprocessor.feature_names_in_)
    missing = [c for c in prep_cols if c not in train_df.columns]
    if missing:
        raise ValueError(f"Missing preprocessor input columns: {missing}")
    feature_cols = [c for c in prep_cols if c not in drop]
else:
    feature_cols = [c for c in train_df.columns if c not in drop]

X = train_df[feature_cols]
y = train_df[targets[0]].to_numpy()

n_out = preprocessor.transform(X.head(5)).shape[1]
k = min(20, n_out)
print(f"n_feature_cols={len(feature_cols)} preproc_out={n_out} k={k}")

selector_pipe = Pipeline([
    ('preprocessing', preprocessor),
    ('selectkbest', SelectKBest(partial(mutual_info_classif, random_state=42), k=k)),
])

fitted = mlca_runtime.fit_record(selector_pipe, X, 'train', y=y)
mlca_runtime.save_model(fitted, 'selector.pkl')

X_sel = fitted.transform(X.head(5))
print(f"selected_sample_shape={X_sel.shape}")
mlca_runtime.record_metric('n_features_selected', float(k), split='train')
