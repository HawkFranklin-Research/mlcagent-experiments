import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())

for key in ['targets', 'excluded_columns', 'group_column', 'task']:
    if key not in protocol:
        raise KeyError(f"Missing {key}. Available: {list(protocol.keys())}")

targets = protocol['targets']
excluded = protocol['excluded_columns']
group_col = protocol.get('group_column')

train = pd.read_parquet(inputs['train'])

all_cols = list(train.columns)
feature_cols = []
for c in all_cols:
    if c in targets or c in excluded or c == group_col:
        continue
    s = train[c]
    if s.nunique(dropna=False) == len(train):
        if pd.api.types.is_float_dtype(s):
            feature_cols.append(c)
        elif pd.api.types.is_integer_dtype(s):
            continue
        else:
            continue
    else:
        feature_cols.append(c)

X = train[feature_cols]
y = train[targets]

preprocessor = pickle.load(open(inputs['05_preprocess_fit_preprocessing_XYF9R0M6/preprocessor.pkl'], 'rb'))
selector = pickle.load(open(inputs['06_features_retain_features_P9FZ0GCB/selector_all.pkl'], 'rb'))
model = pickle.load(open(inputs['07_model_train_baseline_SGXK026F/model.pkl'], 'rb'))

def get_frontend_steps(model, selector, preprocessor):
    if isinstance(model, Pipeline):
        steps = list(model.named_steps.items())
        if steps and hasattr(steps[-1][1], 'predict'):
            return steps[:-1]
        return steps
    if isinstance(selector, Pipeline):
        return list(selector.named_steps.items())
    return [('preprocessor', preprocessor), ('selector', selector)]

frontend_steps = get_frontend_steps(model, selector, preprocessor)
new_pipeline = Pipeline(frontend_steps + [('model', MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42)))])

mlca_runtime.fit_record(new_pipeline, X, 'train', y=y)
mlca_runtime.save_model(new_pipeline, 'alternative_model.pkl')

print(f"Train shape: {train.shape}")
print(f"Features: {len(feature_cols)}")
print(f"Targets: {targets}")
print("Saved alternative_model.pkl")
