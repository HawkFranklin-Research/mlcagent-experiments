import mlca_runtime
import pandas as pd
import json
import pickle
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())

targets = protocol['targets']
excluded = protocol['excluded_columns']
group = protocol.get('group_column')

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

with open(inputs['05_preprocess_fit_preprocessing_ZR2ARXZ1/preprocessor.pkl'], 'rb') as f:
    preprocessor = pickle.load(f)
with open(inputs['06_features_retain_features_H6GS4FCS/selector_all.pkl'], 'rb') as f:
    selector = pickle.load(f)
with open(inputs['07_model_train_baseline_7Z79BW5N/model.pkl'], 'rb') as f:
    model_baseline = pickle.load(f)
with open(inputs['08_model_train_alternative_PY0Z053P/alternative_model.pkl'], 'rb') as f:
    model_alt = pickle.load(f)

def is_complete_pipeline(obj):
    return isinstance(obj, Pipeline) and len(obj.named_steps) > 1

def build_full_pipeline(preprocessor, selector, model):
    if is_complete_pipeline(model):
        return model
    if is_complete_pipeline(selector):
        return Pipeline([('selector', selector), ('model', model)])
    return Pipeline([('preprocessor', preprocessor), ('selector', selector), ('model', model)])

def get_expected_columns(pipeline):
    if isinstance(pipeline, Pipeline):
        first_step = pipeline.steps[0][1]
        if hasattr(first_step, 'feature_names_in_'):
            return list(first_step.feature_names_in_)
    if hasattr(pipeline, 'feature_names_in_'):
        return list(pipeline.feature_names_in_)
    return None

all_cols = train.columns.tolist()
feature_cols = [c for c in all_cols if c not in targets and c not in excluded and c != group]

pipelines = {
    'baseline': build_full_pipeline(preprocessor, selector, model_baseline),
    'alternative': build_full_pipeline(preprocessor, selector, model_alt),
}

preds = {}
for name, pipe in pipelines.items():
    expected = get_expected_columns(pipe)
    if expected is None:
        expected = feature_cols
    missing = [c for c in expected if c not in val.columns]
    if missing:
        raise ValueError(f"Missing columns for {name}: {missing}")
    X_val = val[expected].copy()
    preds[name] = pipe.predict(X_val)

out = pd.DataFrame({
    'row_id': val.index,
    'partition': 'val',
})
for t in targets:
    out[t] = val[t].values
for name, p in preds.items():
    out[f'pred_{name}'] = p

mlca_runtime.save_row_outputs(out, "validation_predictions.parquet")
print(f"Saved validation predictions: {out.shape}")
