import mlca_runtime
import pandas as pd
import numpy as np
import json

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol_path = inputs['_protocol']
data_path = inputs['rna_pr']

with open(protocol_path) as f:
    protocol = json.load(f)

required = ['targets', 'excluded_columns', 'group_column', 'task']
for k in required:
    if k not in protocol:
        raise KeyError(f"Missing key '{k}' in _protocol. Available keys: {list(protocol.keys())}")

targets = protocol['targets']
excluded = protocol['excluded_columns']
group_col = protocol.get('group_column')
task = protocol['task']

df = pd.read_parquet(data_path)

feature_cols = [c for c in df.columns if c not in targets and c not in excluded and c != group_col]
drop_cols = []
for c in feature_cols:
    if df[c].dtype == 'object' or pd.api.types.is_string_dtype(df[c]):
        if df[c].nunique() == len(df):
            drop_cols.append(c)
    elif pd.api.types.is_integer_dtype(df[c]):
        if df[c].nunique() == len(df) and 'id' in c.lower():
            drop_cols.append(c)
feature_cols = [c for c in feature_cols if c not in drop_cols]

for c in feature_cols:
    if pd.api.types.is_datetime64_any_dtype(df[c]):
        df[c] = df[c].astype('int64') // 10**9

numeric_cols = []
cat_cols = []
for c in feature_cols:
    if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
        numeric_cols.append(c)
    else:
        cat_cols.append(c)

target_col = targets[0]
classes = df[target_col].dropna().unique()
if len(classes) > 5:
    top_classes = df[target_col].value_counts().nlargest(5).index.tolist()
else:
    top_classes = classes.tolist()

groups = [('overall', df)] + [(f'{target_col}={cls}', df[df[target_col] == cls]) for cls in top_classes]

rows = []
for group_name, subdf in groups:
    if numeric_cols:
        desc = subdf[numeric_cols].describe().T
        desc['n_missing'] = len(subdf) - desc['count']
        desc['sd'] = desc['std']
        desc['median'] = desc['50%']
        for stat in ['mean', 'sd', 'min', 'median', 'max', 'n_missing']:
            vals = desc[stat].dropna()
            if len(vals) == 0:
                continue
            rows.append({
                'group': group_name,
                'type': 'numeric',
                'n_columns': len(numeric_cols),
                'stat': stat,
                'mean': vals.mean(),
                'sd': vals.std(),
                'min': vals.min(),
                'max': vals.max()
            })
    if cat_cols:
        cat_data = []
        for c in cat_cols:
            s = subdf[c].dropna()
            n_levels = s.nunique()
            top_share = s.value_counts(normalize=True).iloc[0] if len(s) > 0 else np.nan
            cat_data.append({'n_levels': n_levels, 'top_share': top_share})
        cat_df = pd.DataFrame(cat_data)
        for stat in ['n_levels', 'top_share']:
            vals = cat_df[stat].dropna()
            if len(vals) == 0:
                continue
            rows.append({
                'group': group_name,
                'type': 'categorical',
                'n_columns': len(cat_cols),
                'stat': stat,
                'mean': vals.mean(),
                'sd': vals.std(),
                'min': vals.min(),
                'max': vals.max()
            })

result = pd.DataFrame(rows)

def round_sig(x, sig=4):
    if pd.isna(x):
        return x
    return float(f"{x:.{sig}g}")

for col in ['mean', 'sd', 'min', 'max']:
    if col in result.columns:
        result[col] = result[col].apply(round_sig)

mlca_runtime.save_table(result, "distributions.csv", caption="Distribution summary of numeric and categorical features by target class")

n_numeric = len(numeric_cols)
n_categorical = len(cat_cols)
mlca_runtime.record_fact_candidate('n_numeric', n_numeric, source='distributions.csv')
mlca_runtime.record_fact_candidate('n_categorical', n_categorical, source='distributions.csv')

print(f"Features: {n_numeric} numeric, {n_categorical} categorical")
print("Saved distributions.csv")
