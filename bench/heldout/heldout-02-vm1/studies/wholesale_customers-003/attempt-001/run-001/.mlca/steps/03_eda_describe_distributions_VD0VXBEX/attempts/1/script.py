import mlca_runtime
import pandas as pd
import numpy as np
import json

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
protocol = json.loads(inputs["_protocol"].read_text())
try:
    targets = protocol['targets']
    excluded = protocol['excluded_columns']
    group_col = protocol['group_column']
    task = protocol['task']
except KeyError as e:
    raise KeyError(f"Missing key {e} in _protocol. Available keys: {list(protocol.keys())}")

df = pd.read_parquet(inputs["wholesale_customers_uci"])
feature_cols = [c for c in df.columns if c not in targets and c not in excluded and c != group_col]

# drop key-like columns
for c in feature_cols[:]:
    s = df[c]
    if (s.dtype == object or s.dtype.name == 'string') and s.nunique() == len(df):
        feature_cols.remove(c)
    elif pd.api.types.is_integer_dtype(s) and s.nunique() == len(df):
        ss = s.sort_values()
        if (ss.diff().dropna() == 1).all():
            feature_cols.remove(c)

# placeholder handling for text columns
placeholders = ['?', '-', 'NA', 'N/A', '']
placeholder_cols = []
for c in feature_cols:
    if df[c].dtype == object or df[c].dtype.name == 'string':
        df[c] = df[c].astype(str).str.strip()
        mask = df[c].isin(placeholders) | (df[c] == '')
        if mask.any():
            df[c] = df[c].replace(placeholders, pd.NA).replace('', pd.NA)
            placeholder_cols.append(c)
if placeholder_cols:
    print(f"Placeholders found in {len(placeholder_cols)} columns: {placeholder_cols}")

rows = []
for c in feature_cols:
    s = df[c]
    n_missing = int(s.isna().sum())
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        if s.notna().any():
            mean, sd, minv, median, maxv = s.mean(), s.std(), s.min(), s.median(), s.max()
        else:
            mean = sd = minv = median = maxv = np.nan
        rows.append({'feature': c, 'type': 'numeric', 'mean': mean, 'sd': sd,
                     'min': minv, 'median': median, 'max': maxv,
                     'n_missing': n_missing, 'n_levels': np.nan, 'top_share': np.nan})
    else:
        n_levels = s.nunique(dropna=True)
        top_share = s.value_counts(normalize=True, dropna=True).iloc[0] if n_levels else np.nan
        rows.append({'feature': c, 'type': 'categorical', 'mean': np.nan, 'sd': np.nan,
                     'min': np.nan, 'median': np.nan, 'max': np.nan,
                     'n_missing': n_missing, 'n_levels': n_levels, 'top_share': top_share})

summary = pd.DataFrame(rows)
summary = summary.round(4)
mlca_runtime.save_table(summary, "distributions.csv", caption="Summary statistics of feature distributions")

n_numeric = int((summary['type'] == 'numeric').sum())
n_categorical = int((summary['type'] == 'categorical').sum())
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")

print(f"data shape: {df.shape}")
print(f"features summarized: {len(feature_cols)}")
print(f"n_numeric={n_numeric}, n_categorical={n_categorical}")
