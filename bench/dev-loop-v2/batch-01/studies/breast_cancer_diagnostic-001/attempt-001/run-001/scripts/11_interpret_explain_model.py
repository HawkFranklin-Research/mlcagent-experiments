import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
targets = protocol['targets']
excluded = protocol['excluded_columns']
group_col = protocol.get('group_column')
primary_metric = protocol['task']['primary_metric']

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])
print(f"train rows: {len(train)}, val rows: {len(val)}")

with open(inputs['05_preprocess_fit_preprocessing_6GF5BX9F/preprocessor.pkl'], 'rb') as f:
    preprocessor = pickle.load(f)
with open(inputs['06_features_select_features_1B1C3SBH/selector.pkl'], 'rb') as f:
    selector = pickle.load(f)
with open(inputs['07_model_train_logistic_2M248THN/model.pkl'], 'rb') as f:
    model = pickle.load(f)

raw_cols = [c for c in train.columns
            if c not in targets and c not in excluded and c != group_col]

def accepts_raw(obj):
    fn = getattr(obj, 'feature_names_in_', None)
    if fn is None:
        return False
    try:
        return set(fn).issubset(set(train.columns))
    except Exception:
        return False

if accepts_raw(model):
    pipe = model
    feature_cols = list(model.feature_names_in_)
elif isinstance(selector, Pipeline) and accepts_raw(selector):
    pipe = Pipeline([('selector', selector), ('model', model)])
    feature_cols = list(selector.feature_names_in_)
elif accepts_raw(selector):
    pipe = Pipeline([('selector', selector), ('model', model)])
    feature_cols = list(selector.feature_names_in_)
else:
    pipe = Pipeline([('preprocessor', preprocessor),
                     ('selector', selector),
                     ('model', model)])
    feature_cols = raw_cols

X_val = val[feature_cols].copy()
y_val = val[targets[0]].copy()

last = pipe.named_steps[list(pipe.named_steps)[-1]] if isinstance(pipe, Pipeline) else pipe
classes = getattr(last, 'classes_', None)
if classes is not None and len(classes) == 2:
    y_val = y_val.map({c: i for i, c in enumerate(classes)})

scoring = 'roc_auc' if primary_metric == 'roc_auc' else 'roc_auc_ovr'
result = permutation_importance(pipe, X_val, y_val, scoring=scoring,
                                n_repeats=20, random_state=42, n_jobs=1)

imp_df = pd.DataFrame({
    'feature': feature_cols,
    'importance_mean': result.importances_mean,
    'importance_sd': result.importances_std,
}).sort_values('importance_mean', ascending=False).head(15).reset_index(drop=True)
imp_df['importance_mean'] = imp_df['importance_mean'].round(4)
imp_df['importance_sd'] = imp_df['importance_sd'].round(4)

mlca_runtime.save_table(imp_df, 'interpretation.csv',
                        caption='Permutation importance top 15 features')
top_feature = imp_df.iloc[0]['feature']
mlca_runtime.record_fact_candidate('top_feature', top_feature,
                                   source='interpretation.csv')
print(f"top feature: {top_feature}")
