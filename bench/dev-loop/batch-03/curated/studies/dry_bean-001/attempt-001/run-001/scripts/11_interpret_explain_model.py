import mlca_runtime
import json
import pickle
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())

try:
    targets = protocol['targets']
    excluded = protocol.get('excluded_columns', [])
    group = protocol.get('group_column', None)
    task = protocol['task']
except KeyError as e:
    raise KeyError(f"Missing protocol key {e}. Available top-level keys: {list(protocol.keys())}")

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

def is_key_like(s):
    if pd.api.types.is_integer_dtype(s):
        return s.nunique(dropna=False) == len(s)
    if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
        return s.nunique(dropna=False) == len(s)
    return False

feature_cols = [c for c in train.columns if c not in targets and c not in excluded and c != group and not is_key_like(train[c])]
feature_cols = [c for c in feature_cols if c in val.columns]

with open(inputs['latest_model'], 'rb') as f:
    model = pickle.load(f)
if not isinstance(model, Pipeline):
    with open(inputs['07_model_train_logistic_SNK2GEND/model.pkl'], 'rb') as f:
        model = pickle.load(f)

n_classes = task.get('n_classes', len(val[targets[0]].unique()))
scoring = 'roc_auc_ovr' if n_classes > 2 else 'roc_auc'

X_val = val[feature_cols]
y_val = val[targets[0]]

result = permutation_importance(
    model, X_val, y_val, scoring=scoring, n_repeats=20, random_state=42, n_jobs=1
)

df_imp = pd.DataFrame({
    'feature': feature_cols,
    'importance_mean': result.importances_mean,
    'importance_sd': result.importances_std
})
df_imp = df_imp.sort_values('importance_mean', ascending=False).head(15)
df_imp['importance_mean'] = df_imp['importance_mean'].apply(lambda x: float(f'{x:.4g}'))
df_imp['importance_sd'] = df_imp['importance_sd'].apply(lambda x: float(f'{x:.4g}'))

mlca_runtime.save_table(df_imp, "interpretation.csv", caption="Permutation importance top 15 features")
top_feature = df_imp.iloc[0]['feature']
mlca_runtime.record_fact_candidate('top_feature', top_feature, source='interpretation.csv')

print(f"Validation rows: {len(val)}")
print(f"Features used: {len(feature_cols)}")
print(f"Top feature: {top_feature}")
print(f"Top importance mean: {df_imp.iloc[0]['importance_mean']}")
