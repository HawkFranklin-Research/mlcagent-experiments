import mlca_runtime
import pandas as pd
import numpy as np
import json

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
heart_disease_path = inputs['heart_disease_cleveland_uci']
protocol_path = inputs['_protocol']

with open(protocol_path, 'r') as f:
    proto = json.load(f)

excluded_columns = proto.get('excluded_columns', [])

df = pd.read_parquet(heart_disease_path)
df = df.drop(columns=excluded_columns, errors='ignore')

target_col = 'heart_disease_binary'
if target_col not in df.columns:
    raise ValueError(f"Target column {target_col} not found in data")

numeric_cols = [c for c in df.columns if df[c].dtype.kind in 'biufc' and c != target_col]
categorical_cols = [c for c in df.columns if c not in numeric_cols and c != target_col]

n_numeric = len(numeric_cols)
n_categorical = len(categorical_cols)

rows = []

def add_numeric_row(col, cls):
    series = df[col]
    if cls == 'overall':
        mean = series.mean(skipna=True)
        sd = series.std(skipna=True)
        mn = series.min(skipna=True)
        med = series.median(skipna=True)
        mx = series.max(skipna=True)
        nm = series.isna().sum()
    elif cls == '0':
        mask = df[target_col] == 0
        sub = series[mask]
        if sub.empty:
            mean = sd = mn = med = mx = nm = np.nan
        else:
            mean = sub.mean(skipna=True)
            sd = sub.std(skipna=True)
            mn = sub.min(skipna=True)
            med = sub.median(skipna=True)
            mx = sub.max(skipna=True)
            nm = sub.isna().sum()
    else:
        mask = df[target_col] == 1
        sub = series[mask]
        if sub.empty:
            mean = sd = mn = med = mx = nm = np.nan
        else:
            mean = sub.mean(skipna=True)
            sd = sub.std(skipna=True)
            mn = sub.min(skipna=True)
            med = sub.median(skipna=True)
            mx = sub.max(skipna=True)
            nm = sub.isna().sum()
    rows.append({
        'column': col,
        'class': cls,
        'mean': mean,
        'sd': sd,
        'min': mn,
        'median': med,
        'max': mx,
        'n_missing': nm,
        'n_levels': np.nan,
        'top_level_share': np.nan
    })

for col in numeric_cols:
    add_numeric_row(col, 'overall')
    add_numeric_row(col, '0')
    add_numeric_row(col, '1')

for col in categorical_cols:
    series = df[col]
    unique_vals = series.dropna().unique()
    n_levels = len(unique_vals)
    if len(unique_vals) > 0:
        counts = series.value_counts(dropna=True)
        top_share = counts.iloc[0] / counts.sum()
    else:
        top_share = 0.0
    rows.append({
        'column': col,
        'class': 'overall',
        'mean': np.nan,
        'sd': np.nan,
        'min': np.nan,
        'median': np.nan,
        'max': np.nan,
        'n_missing': series.isna().sum(),
        'n_levels': n_levels,
        'top_level_share': top_share
    })

dist_df = pd.DataFrame(rows)

for col in ['mean', 'sd', 'min', 'median', 'max', 'n_missing', 'top_level_share']:
    dist_df[col] = dist_df[col].round(4)

dist_path = mlca_runtime.save_table(dist_df, name='distributions.csv', caption='Distributions summary')

mlca_runtime.record_fact_candidate('n_numeric', n_numeric, source='distributions.csv')
mlca_runtime.record_fact_candidate('n_categorical', n_categorical, source='distributions.csv')

X = df.drop(columns=[target_col])
y = df[target_col]

print(f"n_numeric: {n_numeric}")
print(f"n_categorical: {n_categorical}")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
