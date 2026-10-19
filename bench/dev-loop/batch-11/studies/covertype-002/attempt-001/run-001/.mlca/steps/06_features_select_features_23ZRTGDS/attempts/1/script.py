import json, pickle
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

if isinstance(preprocessor, Pipeline):
    print(f"preprocessor pipeline steps: {list(preprocessor.named_steps.keys())}")

if hasattr(preprocessor, 'feature_names_in_'):
    feature_cols = [c for c in preprocessor.feature_names_in_
                    if c not in targets and c not in excluded and c != group_col]
else:
    drop = set(targets) | set(excluded) | ({group_col} if group_col else set())
    feature_cols = [c for c in train_df.columns if c not in drop]

X = train_df[feature_cols].copy()
y = train_df[targets[0]].values

n_out = preprocessor.transform(X.head(5)).shape[1]
k = min(20, n_out)
print(f"n_feature_cols={len(feature_cols)} preprocessor_out={n_out} k={k}")

selector_pipe = Pipeline([
    ('preprocessing', preprocessor),
    ('selectkbest', SelectKBest(mutual_info_classif, k=k, random_state=42)),
])

fitted = mlca_runtime.fit_record(selector_pipe, X, 'train', y=y)
mlca_runtime.save_model(fitted, 'selector.pkl')

X_sel = fitted.transform(X.head(5))
print(f"selected_sample_shape={X_sel.shape}")
names = fitted.named_steps['selectkbest'].get_feature_names_out()
print(f"first_selected_feature={names[0]}")
mlca_runtime.record_metric('n_features_selected', float(k), split='train')
