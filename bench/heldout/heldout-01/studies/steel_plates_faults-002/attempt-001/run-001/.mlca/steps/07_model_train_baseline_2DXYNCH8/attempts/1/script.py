import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
targets = protocol['targets']
excluded = protocol['excluded_columns']
group_col = protocol.get('group_column', None)

train_df = pd.read_parquet(inputs['train'])
val_df = pd.read_parquet(inputs['val'])
y_train = train_df[targets]
y_val = val_df[targets]

with open(inputs['05_preprocess_fit_preprocessing_18P9EMFB/preprocessor.pkl'], 'rb') as f:
    preprocessor = pickle.load(f)
with open(inputs['06_features_retain_features_ZDXE7YSF/selector_all.pkl'], 'rb') as f:
    selector = pickle.load(f)

base = None
if isinstance(selector, Pipeline):
    has_preproc = any(
        isinstance(step, ColumnTransformer) or (isinstance(step, Pipeline) and any(isinstance(s, ColumnTransformer) for s in step.named_steps.values()))
        for step in selector.named_steps.values()
    )
    if has_preproc or 'preprocessor' in selector.named_steps:
        base = selector
    elif hasattr(selector, 'feature_names_in_') and set(selector.feature_names_in_).issubset(set(train_df.columns)):
        base = selector
else:
    if hasattr(selector, 'feature_names_in_') and set(selector.feature_names_in_).issubset(set(train_df.columns)):
        base = selector

clf = MultiOutputClassifier(LogisticRegression(max_iter=2000))
if base is not None:
    final_pipeline = Pipeline([('base', base), ('classifier', clf)])
    input_cols = list(base.feature_names_in_) if hasattr(base, 'feature_names_in_') else [c for c in train_df.columns if c not in targets and c not in excluded and c != group_col]
else:
    final_pipeline = Pipeline([('preprocessor', preprocessor), ('selector', selector), ('classifier', clf)])
    input_cols = list(preprocessor.feature_names_in_) if hasattr(preprocessor, 'feature_names_in_') else [c for c in train_df.columns if c not in targets and c not in excluded and c != group_col]

input_cols = [c for c in input_cols if c not in targets and c not in excluded]
X_train = train_df[input_cols]
X_val = val_df[input_cols]

mlca_runtime.fit_record(final_pipeline, X_train, 'train', y=y_train)
mlca_runtime.save_model(final_pipeline, "model.pkl")
print(f"Fitted pipeline with {len(input_cols)} features. Train shape: {X_train.shape}")
