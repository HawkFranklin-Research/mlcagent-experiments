import mlca_runtime
import pandas as pd
import numpy as np
import json

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
df = pd.read_parquet(inputs['wine_quality_uci'])

targets = protocol['targets']
excluded = protocol.get('excluded_columns', [])
group_col = protocol.get('group_column', None)

# RL-0007: feature columns
all_cols = df.columns.tolist()
feature_cols = [c for c in all_cols if c not in targets and c not in excluded and c != group_col]

# Drop key-like text columns (all distinct)
final_features = []
for c in feature_cols:
    s = df[c]
    if pd.api.types.is_string_dtype(s) or s.dtype == 'object':
        if s.nunique() == len(s) and s.nunique() > 1:
            continue
    final_features.append(c)

# RL-0023: placeholder check
placeholder_cols = 0
for c in final_features:
    s = df[c]
    if pd.api.types.is_string_dtype(s) or s.dtype == 'object':
        vals = s.dropna().astype(str).str.strip()
        placeholders = {'?', '-', 'NA', 'N/A', ''}
        if vals.isin(placeholders).any() or vals.str.match(r'^\s*$').any():
            placeholder_cols += 1
print(f"Placeholder columns found: {placeholder_cols}")

# RL-0010: classify types
numeric_cols = []
categorical_cols = []
datetime_cols = []
for c in final_features:
    s = df[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        datetime_cols.append(c)
    elif pd.api.types.is_numeric_dtype(s):
        numeric_cols.append(c)
    else:
        if s.nunique() / len(s) > 0.05 and s.nunique() > 10:
            pass
        else:
            categorical_cols.append(c)

# Derive numeric parts from datetime columns
for c in datetime_cols:
    s = pd.to_datetime(df[c])
    for part, func in [('hour', lambda x: x.dt.hour),
                       ('dayofweek', lambda x: x.dt.dayofweek),
                       ('month', lambda x: x.dt.month)]:
        newc = f"{c}_{part}"
        df[newc] = func(s)
        numeric_cols.append(newc)

# Build summary rows
rows = []
for c in numeric_cols:
    s = df[c]
    rows.append({
        'variable': c, 'type': 'numeric', 'group': 'overall',
        'n': len(s), 'n_missing': s.isna().sum(),
        'mean': s.mean(), 'sd': s.std(), 'min': s.min(),
        'median': s.median(), 'max': s.max(),
        'n_levels': np.nan, 'top_share': np.nan
    })
for c in categorical_cols:
    s = df[c]
    vc = s.value_counts(dropna=True)
    n_levels = s.nunique()
    top_share = vc.iloc[0] / len(s) if n_levels > 0 else np.nan
    rows.append({
        'variable': c, 'type': 'categorical', 'group': 'overall',
        'n': len(s), 'n_missing': s.isna().sum(),
        'mean': np.nan, 'sd': np.nan, 'min': np.nan,
        'median': np.nan, 'max': np.nan,
        'n_levels': n_levels, 'top_share': top_share
    })

# Per-class rows for numeric columns (limit to 50 rows total)
max_rows = 50
current_rows = len(rows)
if targets and targets[0] in df.columns and current_rows < max_rows:
    target_col = targets[0]
    classes = df[target_col].dropna().unique()
    remaining = max_rows - current_rows
    if len(classes) > 0:
        k = remaining // len(classes)
        k = min(k, len(numeric_cols))
        for cls in classes:
            sub = df[df[target_col] == cls]
            for c in numeric_cols[:k]:
                s = sub[c]
                rows.append({
                    'variable': c, 'type': 'numeric', 'group': str(cls),
                    'n': len(s), 'n_missing': s.isna().sum(),
                    'mean': s.mean(), 'sd': s.std(), 'min': s.min(),
                    'median': s.median(), 'max': s.max(),
                    'n_levels': np.nan, 'top_share': np.nan
                })

summary = pd.DataFrame(rows)
for col in ['mean','sd','min','median','max','top_share']:
    if col in summary.columns:
        summary[col] = summary[col].apply(
            lambda x: float(f"{x:.4g}") if pd.notna(x) and isinstance(x, (int, float)) else x
        )

mlca_runtime.save_table(summary, 'distributions.csv', caption='Distribution summary of features')

n_numeric = len(numeric_cols)
n_categorical = len(categorical_cols)
mlca_runtime.record_fact_candidate('n_numeric', n_numeric, source='distributions.csv')
mlca_runtime.record_fact_candidate('n_categorical', n_categorical, source='distributions.csv')

print(f"Features: {len(final_features)} (numeric={n_numeric}, categorical={n_categorical})")
print(f"Summary rows: {len(summary)}")
