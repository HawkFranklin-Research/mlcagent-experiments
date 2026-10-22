import mlca_runtime
import json
import pandas as pd
import numpy as np

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())

required = ['targets', 'excluded_columns']
missing = [k for k in required if k not in protocol]
if missing:
    raise KeyError(f"Missing required protocol keys: {missing}. Available keys: {list(protocol.keys())}")

targets = protocol['targets']
excluded = protocol['excluded_columns']
group_col = protocol.get('group_column', None)

df = pd.read_parquet(inputs['bike_sharing_hourly_uci'])

exclude_set = set(targets) | set(excluded)
if group_col is not None:
    exclude_set.add(group_col)

feature_cols = []
for col in df.columns:
    if col in exclude_set:
        continue
    s = df[col]
    if (pd.api.types.is_integer_dtype(s) or pd.api.types.is_object_dtype(s)) and s.nunique() == len(df):
        continue
    feature_cols.append(col)

rows = []
for col in feature_cols:
    s = df[col]
    n_missing = s.isna().sum()
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        typ = 'numeric'
        mean = s.mean()
        sd = s.std()
        minv = s.min()
        median = s.median()
        maxv = s.max()
        n_levels = np.nan
        top_share = np.nan
    else:
        typ = 'categorical'
        mean = sd = minv = median = maxv = np.nan
        n_levels = s.nunique(dropna=True)
        top_share = s.value_counts(normalize=True, dropna=True).iloc[0] if n_levels > 0 else np.nan
    rows.append({
        'variable': col, 'type': typ, 'mean': mean, 'sd': sd, 'min': minv,
        'median': median, 'max': maxv, 'n_missing': n_missing,
        'n_levels': n_levels, 'top_share': top_share
    })

summary_df = pd.DataFrame(rows)
for c in ['mean','sd','min','median','max','top_share']:
    if c in summary_df.columns:
        summary_df[c] = summary_df[c].round(4)

mlca_runtime.save_table(summary_df, "distributions.csv", caption="Distribution summary of numeric and categorical features")

n_numeric = int((summary_df['type'] == 'numeric').sum())
n_categorical = int((summary_df['type'] == 'categorical').sum())
mlca_runtime.record_fact_candidate('n_numeric', n_numeric, source='distributions.csv')
mlca_runtime.record_fact_candidate('n_categorical', n_categorical, source='distributions.csv')

print(f"Saved distributions.csv: {len(summary_df)} rows, {n_numeric} numeric, {n_categorical} categorical")
