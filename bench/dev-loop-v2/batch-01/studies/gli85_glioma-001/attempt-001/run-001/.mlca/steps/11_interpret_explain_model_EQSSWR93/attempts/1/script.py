import mlca_runtime
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance
from math import floor, log10

def round_sig(x, sig=4):
    if x == 0:
        return 0.0
    return round(x, -int(floor(log10(abs(x)))) + (sig - 1))

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())

targets = protocol.get('targets')
if targets is None:
    raise KeyError(f"Missing 'targets' in _protocol. Available keys: {list(protocol.keys())}")
excluded = protocol.get('excluded_columns', [])
group_col = protocol.get('group_column')
task = protocol.get('task', {})
primary_metric = task.get('primary_metric', 'roc_auc')

train = pd.read_parquet(inputs['train'])
val = pd.read_parquet(inputs['val'])

with open(inputs['latest_model'], 'rb') as f:
    model = pickle.load(f)

if hasattr(model, 'feature_names_in_'):
    feature_cols = list(model.feature_names_in_)
else:
    all_cols = train.columns.tolist()
    feature_cols = [c for c in all_cols if c not in targets and c not in excluded and c != group_col]
    for c in list(feature_cols):
        if train[c].dtype == 'object' and train[c].nunique() == len(train):
            feature_cols.remove(c)
        elif np.issubdtype(train[c].dtype, np.integer) and train[c].nunique() == len(train) and train[c].min() >= 0 and train[c].max() < 10*len(train):
            feature_cols.remove(c)

feature_cols = [c for c in feature_cols if c in val.columns]
X_val = val[feature_cols].copy()
y_val = val[targets[0]].copy()

scoring = 'roc_auc' if primary_metric == 'roc_auc' else 'roc_auc_ovr'
result = permutation_importance(model, X_val, y_val, n_repeats=20, random_state=42, scoring=scoring, n_jobs=1)

imp_df = pd.DataFrame({
    'feature': feature_cols,
    'importance_mean': result.importances_mean,
    'importance_sd': result.importances_std
})
imp_df = imp_df.sort_values('importance_mean', ascending=False).head(15).reset_index(drop=True)
imp_df['importance_mean'] = imp_df['importance_mean'].apply(round_sig)
imp_df['importance_sd'] = imp_df['importance_sd'].apply(round_sig)

mlca_runtime.save_table(imp_df, 'interpretation.csv', caption='Permutation importance on validation set')
top_feature = imp_df.iloc[0]['feature']
mlca_runtime.record_fact_candidate('top_feature', top_feature, source='interpretation.csv')

print(f"Train rows: {len(train)}, Val rows: {len(val)}")
print(f"Features used: {len(feature_cols)}")
print(f"Top feature: {top_feature} (mean importance: {imp_df.iloc[0]['importance_mean']})")
