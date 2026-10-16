import mlca_runtime
import json
import pandas as pd
import numpy as np

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())

required = ['targets', 'excluded_columns', 'group_column', 'task']
missing = [k for k in required if k not in protocol]
if missing:
    raise KeyError(f"Missing protocol keys: {missing}. Available: {list(protocol.keys())}")

targets = protocol['targets']
excluded = protocol['excluded_columns']
group_col = protocol['group_column']

df = pd.read_parquet(inputs['optical_digits_uci'])

key_like = []
for col in df.columns:
    if col in targets or col in excluded or col == group_col:
        continue
    if df[col].dtype == 'object' or df[col].dtype.name == 'string':
        if df[col].nunique(dropna=True) == len(df) and df[col].notna().all():
            key_like.append(col)
    elif pd.api.types.is_integer_dtype(df[col]):
        if df[col].nunique(dropna=True) == len(df) and df[col].notna().all():
            key_like.append(col)

feature_cols = [c for c in df.columns if c not in targets and c not in excluded and c != group_col and c not in key_like]

numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
categorical_cols = [c for c in feature_cols if c not in numeric_cols]

text_cols = [c for c in feature_cols if df[c].dtype == 'object' or df[c].dtype.name == 'string']
placeholders = ['?', '-', 'NA', 'N/A', '']
placeholder_counts = {}
for col in text_cols:
    s = df[col].astype(str).str.strip()
    found = set()
    for ph in placeholders:
        if (s == ph).any():
            found.add(ph)
    if found:
        placeholder_counts[col] = found
if placeholder_counts:
    print(f"Placeholders found in {len(placeholder_counts)} columns: {placeholder_counts}")
else:
    print("No placeholders found in text columns.")

num_stats_list = []
for col in numeric_cols:
    s = df[col]
    num_stats_list.append({
        'feature': col,
        'mean': s.mean(),
        'sd': s.std(),
        'min': s.min(),
        'median': s.median(),
        'max': s.max(),
        'n_missing': s.isna().sum()
    })
num_stats = pd.DataFrame(num_stats_list)

cat_stats_list = []
for col in categorical_cols:
    s = df[col]
    vc = s.value_counts(dropna=True)
    n_levels = s.nunique(dropna=True)
    top_share = vc.iloc[0] / len(s) if len(vc) > 0 else np.nan
    cat_stats_list.append({
        'feature': col,
        'n_levels': n_levels,
        'top_level_share': top_share
    })
cat_stats = pd.DataFrame(cat_stats_list)

rows = []
for stat in ['mean', 'sd', 'min', 'median', 'max', 'n_missing']:
    if len(num_stats) > 0:
        vals = num_stats[stat]
        rows.append({
            'column_type': 'numeric',
            'statistic': stat,
            'mean': vals.mean(),
            'sd': vals.std(),
            'min': vals.min(),
            'median': vals.median(),
            'max': vals.max()
        })
    else:
        rows.append({
            'column_type': 'numeric',
            'statistic': stat,
            'mean': np.nan,
            'sd': np.nan,
            'min': np.nan,
            'median': np.nan,
            'max': np.nan
        })

for stat in ['n_levels', 'top_level_share']:
    if len(cat_stats) > 0:
        vals = cat_stats[stat]
        rows.append({
            'column_type': 'categorical',
            'statistic': stat,
            'mean': vals.mean(),
            'sd': vals.std(),
            'min': vals.min(),
            'median': vals.median(),
            'max': vals.max()
        })
    else:
        rows.append({
            'column_type': 'categorical',
            'statistic': stat,
            'mean': np.nan,
            'sd': np.nan,
            'min': np.nan,
            'median': np.nan,
            'max': np.nan
        })

summary_df = pd.DataFrame(rows).round(4)

mlca_runtime.save_table(summary_df, "distributions.csv", caption="Feature distribution summary")

n_numeric = len(numeric_cols)
n_categorical = len(categorical_cols)
mlca_runtime.record_fact_candidate("n_numeric", n_numeric, source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", n_categorical, source="distributions.csv")

print(f"Rows: {len(df)}, Features: {len(feature_cols)}")
print(f"Numeric: {n_numeric}, Categorical: {n_categorical}")
print("Saved distributions.csv")
