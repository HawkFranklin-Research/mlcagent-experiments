import mlca_runtime
import json
import pickle
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inp = mlca_runtime.inputs()

protocol = json.loads(inp['_protocol'].read_text())
for k in ['targets', 'excluded_columns', 'group_column', 'task']:
    if k not in protocol:
        raise KeyError(f"Missing protocol key '{k}'. Available: {list(protocol.keys())}")
targets = protocol['targets']
excluded = protocol['excluded_columns']
group = protocol['group_column']
task = protocol['task']

train = pd.read_parquet(inp['train'])
val = pd.read_parquet(inp['val'])

feature_cols = []
for c in train.columns:
    if c in targets or c in excluded or c == group:
        continue
    col = train[c]
    if (col.dtype == object or pd.api.types.is_string_dtype(col)) and col.nunique() == len(col):
        continue
    if pd.api.types.is_integer_dtype(col) and col.nunique() == len(col):
        continue
    feature_cols.append(c)

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

preprocessor = load_pkl(inp['05_preprocess_fit_preprocessing_BKKYG4P6/preprocessor.pkl'])
selector = load_pkl(inp['06_features_select_features_GQKA7M4V/selector.pkl'])

def expected_cols(obj):
    if hasattr(obj, 'feature_names_in_'):
        return list(obj.feature_names_in_)
    if isinstance(obj, Pipeline) and len(obj.steps) > 0:
        first = obj.steps[0][1]
        if hasattr(first, 'feature_names_in_'):
            return list(first.feature_names_in_)
    return None

def build_predictor(model):
    if isinstance(model, Pipeline) and len(model.named_steps) > 1:
        return model
    steps = []
    if preprocessor is not None and not (isinstance(selector, Pipeline) and any('preprocess' in n.lower() for n in selector.named_steps)):
        steps.append(('preprocessor', preprocessor))
    if selector is not None:
        steps.append(('selector', selector))
    steps.append(('model', model))
    return Pipeline(steps)

cand_keys = [
    '07_model_train_logistic_7WNJSTAG/model.pkl',
    '08_model_train_alternative_84VF3NXG/alternative_model.pkl',
    'latest_model',
]
seen = set()
candidates = []
for key in cand_keys:
    p = Path(inp[key]).resolve()
    if p in seen:
        continue
    seen.add(p)
    model = load_pkl(inp[key])
    predictor = build_predictor(model)
    label = 'logistic' if 'logistic' in key else ('alternative' if 'alternative' in key else key.split('/')[0])
    candidates.append((label, predictor))

pos_label = protocol.get('positive_label') or task.get('positive_label')

X_val = val[feature_cols]
y_val = val[targets[0]]
out = pd.DataFrame(index=val.index)
out['row_id'] = val.index
out['partition'] = 'validation'
out['outcome'] = y_val

for label, predictor in candidates:
    cols = expected_cols(predictor)
    Xm = val[cols] if cols is not None else X_val
    final = predictor.steps[-1][1] if isinstance(predictor, Pipeline) else predictor
    if hasattr(final, 'predict_proba'):
        proba = predictor.predict_proba(Xm)
        classes = list(final.classes_)
        pl = pos_label if pos_label is not None and pos_label in classes else classes[1]
        out[f'pred_{label}'] = proba[:, classes.index(pl)]
    else:
        out[f'pred_{label}'] = predictor.decision_function(Xm)

mlca_runtime.save_row_outputs(out, 'validation_predictions.csv')
print(f"val rows: {len(out)}; features: {len(feature_cols)}; candidates: {[l for l,_ in candidates]}")
