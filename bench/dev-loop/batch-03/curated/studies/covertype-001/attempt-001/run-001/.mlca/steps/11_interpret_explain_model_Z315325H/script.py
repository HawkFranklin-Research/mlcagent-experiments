import mlca_runtime
import json, pickle
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
for k in ['targets', 'excluded_columns', 'group_column', 'task']:
    if k not in protocol:
        raise KeyError(f"Missing {k}. Available: {list(protocol.keys())}")
targets = protocol['targets']
excluded = protocol['excluded_columns']
group = protocol['group_column']
task = protocol['task']
primary_metric = task.get('primary_metric', 'roc_auc_ovr')

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

drop_cols = set(targets) | set(excluded)
if group is not None:
    drop_cols.add(group)
feature_cols = [c for c in train.columns if c not in drop_cols]
keep = []
for c in feature_cols:
    s = train[c]
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        if s.nunique() == len(train):
            continue
    if pd.api.types.is_integer_dtype(s) and s.nunique() == len(train):
        name = c.lower()
        if any(t in name for t in ['id', 'code', 'key', 'index']):
            continue
    keep.append(c)
feature_cols = keep

with open(inputs['latest_model'], 'rb') as f:
    model = pickle.load(f)
if not isinstance(model, Pipeline):
    raise TypeError("Loaded model is not a sklearn Pipeline")
if hasattr(model, 'feature_names_in_'):
    feature_cols = list(model.feature_names_in_)
    if any(c in drop_cols for c in feature_cols):
        raise ValueError("Pipeline expects a target or excluded column")

if len(val) > 10000:
    val_use = val.sample(n=10000, random_state=42)
else:
    val_use = val
print(f"Val rows used: {len(val_use)}")

X_val = val_use[feature_cols]
y_val = val_use[targets[0]]

result = permutation_importance(
    model, X_val, y_val, scoring=primary_metric,
    n_repeats=20, random_state=42, n_jobs=1
)

df = pd.DataFrame({
    'feature': feature_cols,
    'importance_mean': result.importances_mean,
    'importance_sd': result.importances_std
})
df = df.sort_values('importance_mean', ascending=False).head(15).reset_index(drop=True)

def round_sig(x, sig=4):
    if pd.isna(x):
        return x
    return float(f'{x:.{sig}g}')

df['importance_mean'] = df['importance_mean'].apply(round_sig)
df['importance_sd'] = df['importance_sd'].apply(round_sig)

mlca_runtime.save_table(df, 'interpretation.csv', caption='Permutation importance top 15 features')
top_feature = df.iloc[0]['feature']
mlca_runtime.record_fact_candidate('top_feature', top_feature, source='interpretation.csv')
print(f"Top feature: {top_feature}")
print(f"Top importance: {df.iloc[0]['importance_mean']}")
