import mlca_runtime
import json
import pandas as pd
import numpy as np

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol_path = inputs['_protocol']
data_path = inputs['dry_bean_uci']

protocol = json.loads(protocol_path.read_text())

required = ['targets', 'excluded_columns', 'group_column', 'task']
for k in required:
    if k not in protocol:
        raise KeyError(f"Missing key '{k}' in _protocol. Available top-level keys: {list(protocol.keys())}")

targets = protocol['targets']
excluded = protocol['excluded_columns']
group_col = protocol['group_column']
task = protocol['task']

df = pd.read_parquet(data_path)

all_cols = list(df.columns)
feature_cols = []
for col in all_cols:
    if col in targets or col in excluded or col == group_col:
        continue
    if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col]):
        if df[col].nunique(dropna=True) == len(df):
            continue
    if pd.api.types.is_integer_dtype(df[col]):
        if df[col].nunique(dropna=True) == len(df):
            if (df[col].min() == 0 and df[col].max() == len(df) - 1) or any(s in col.lower() for s in ['id', 'key', 'code']):
                continue
    feature_cols.append(col)

placeholders = ['?', '-', 'NA', 'N/A', '']
placeholder_cols = []
for col in feature_cols:
    if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col]):
        s = df[col].astype(str).str.strip()
        if s.isin(placeholders).any():
            placeholder_cols.append(col)
if placeholder_cols:
    print(f"Placeholders found in {len(placeholder_cols)} columns: {placeholder_cols}")
else:
    print("No placeholder tokens found in text columns.")

rows = []
for col in feature_cols:
    s = df[col]
    n_missing = int(s.isna().sum())
    if pd.api.types.is_numeric_dtype(s):
        row = {
            'column': col,
            'type': 'numeric',
            'class': 'ALL',
            'n_missing': n_missing,
            'mean': s.mean(),
            'sd': s.std(),
            'min': s.min(),
            'median': s.median(),
            'max': s.max(),
            'n_levels': np.nan,
            'top_share': np.nan
        }
    else:
        vc = s.value_counts(dropna=True)
        n_levels = int(s.nunique(dropna=True))
        top_share = vc.iloc[0] / len(s) if len(vc) > 0 else np.nan
        row = {
            'column': col,
            'type': 'categorical',
            'class': 'ALL',
            'n_missing': n_missing,
            'mean': np.nan,
            'sd': np.nan,
            'min': np.nan,
            'median': np.nan,
            'max': np.nan,
            'n_levels': n_levels,
            'top_share': top_share
        }
    rows.append(row)

target_col = targets[0]
if target_col in df.columns:
    vc = df[target_col].value_counts(dropna=True)
    for cls, count in vc.items():
        rows.append({
            'column': target_col,
            'type': 'categorical',
            'class': cls,
            'n_missing': np.nan,
            'mean': np.nan,
            'sd': np.nan,
            'min': np.nan,
            'median': np.nan,
            'max': np.nan,
            'n_levels': np.nan,
            'top_share': count / len(df)
        })

dist_df = pd.DataFrame(rows)

for col in ['n_missing', 'mean', 'sd', 'min', 'median', 'max', 'n_levels', 'top_share']:
    if col in dist_df.columns:
        dist_df[col] = dist_df[col].apply(lambda x: float(f"{x:.4g}") if pd.notna(x) and isinstance(x, (int, float, np.number)) else x)

if len(dist_df) > 50:
    dist_df = dist_df.head(50)

mlca_runtime.save_table(dist_df, "distributions.csv", caption="Summary distributions of features and target classes")

n_numeric = sum(1 for col in feature_cols if pd.api.types.is_numeric_dtype(df[col]))
n_categorical = sum(1 for col in feature_cols if not pd.api.types.is_numeric_dtype(df[col]))
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")

print(f"n_numeric={n_numeric}, n_categorical={n_categorical}")
print(f"distributions.csv rows={len(dist_df)}")
