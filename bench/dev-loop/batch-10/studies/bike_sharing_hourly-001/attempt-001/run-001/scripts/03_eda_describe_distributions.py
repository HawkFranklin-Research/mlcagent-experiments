import mlca_runtime
import json
import numpy as np
import pandas as pd

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
df = pd.read_parquet(inputs['bike_sharing_hourly_uci'])

targets = list(protocol['targets'])
excluded = list(protocol['excluded_columns'])
group = protocol.get('group_column')

feature_cols = [c for c in df.columns if c not in set(targets) | set(excluded) and c != group]

# placeholder token scan (RL-0023)
ph_cols = 0
for c in df.columns:
    s = df[c]
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        v = s.dropna().astype(str)
        if v.str.strip().isin(['?', '-', 'NA', 'N/A', '']).any():
            ph_cols += 1
print(f"placeholder_token_columns={ph_cols}")

rows = []
for c in feature_cols:
    s = df[c]
    n_missing = int(s.isna().sum())
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        s2 = s.dropna()
        rows.append({
            'variable': c, 'column_type': 'numeric', 'n_missing': n_missing,
            'mean': float(s2.mean()) if len(s2) else np.nan,
            'sd': float(s2.std()) if len(s2) > 1 else np.nan,
            'min': float(s2.min()) if len(s2) else np.nan,
            'median': float(s2.median()) if len(s2) else np.nan,
            'max': float(s2.max()) if len(s2) else np.nan,
            'n_levels': np.nan, 'top_level_share': np.nan,
        })
    else:
        s2 = s.dropna().astype(str)
        vc = s2.value_counts()
        rows.append({
            'variable': c, 'column_type': 'categorical', 'n_missing': n_missing,
            'mean': np.nan, 'sd': np.nan, 'min': np.nan, 'median': np.nan, 'max': np.nan,
            'n_levels': int(s2.nunique()) if len(s2) else np.nan,
            'top_level_share': float(vc.iloc[0] / len(s2)) if len(vc) else np.nan,
        })

dist = pd.DataFrame(rows)
for col in ['mean', 'sd', 'min', 'median', 'max', 'top_level_share']:
    dist[col] = dist[col].apply(lambda x: float(f'{x:.4g}') if pd.notna(x) else np.nan)
if len(dist) > 50:
    dist = dist.head(50)

mlca_runtime.save_table(dist, 'distributions.csv', caption='Per-column distribution summary')

n_numeric = int((dist['column_type'] == 'numeric').sum())
n_categorical = int((dist['column_type'] == 'categorical').sum())
mlca_runtime.record_fact_candidate('n_numeric', n_numeric, source='distributions.csv')
mlca_runtime.record_fact_candidate('n_categorical', n_categorical, source='distributions.csv')
print(f"features={len(feature_cols)} n_numeric={n_numeric} n_categorical={n_categorical} rows={len(df)}")
