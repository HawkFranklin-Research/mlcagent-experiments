import mlca_runtime
import pandas as pd
import numpy as np
import json

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
df = pd.read_parquet(inputs['bike_sharing_hourly_uci'])

required = ['targets', 'excluded_columns', 'group_column', 'task']
missing = [k for k in required if k not in protocol]
if missing:
    raise ValueError(f"Missing protocol keys: {missing}. Available: {list(protocol.keys())}")

targets = protocol['targets']
excluded = protocol['excluded_columns']
group_col = protocol['group_column']

key_like = []
for c in df.columns:
    if c in targets or c in excluded or c == group_col:
        continue
    if (pd.api.types.is_integer_dtype(df[c]) or pd.api.types.is_object_dtype(df[c])) and df[c].nunique() == len(df):
        key_like.append(c)

feature_cols = [c for c in df.columns if c not in targets and c not in excluded and c != group_col and c not in key_like]

placeholders = ['?', '-', 'NA', 'N/A']
placeholder_cols = []
for c in feature_cols:
    if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]):
        s = df[c].astype(str).str.strip()
        if s.isin(placeholders).any() or (s == '').any():
            placeholder_cols.append(c)
if placeholder_cols:
    print(f"Placeholders found in {len(placeholder_cols)} columns: {placeholder_cols}")
else:
    print("Placeholders: none")

rows = []
for c in feature_cols:
    if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]):
        row = {
            'variable': c,
            'type': 'numeric',
            'group': 'overall',
            'mean': df[c].mean(),
            'sd': df[c].std(),
            'min': df[c].min(),
            'median': df[c].median(),
            'max': df[c].max(),
            'n_missing': df[c].isna().sum(),
            'n_levels': np.nan,
            'top_share': np.nan
        }
    else:
        vc = df[c].value_counts(dropna=True, normalize=True)
        row = {
            'variable': c,
            'type': 'categorical',
            'group': 'overall',
            'mean': np.nan,
            'sd': np.nan,
            'min': np.nan,
            'median': np.nan,
            'max': np.nan,
            'n_missing': df[c].isna().sum(),
            'n_levels': df[c].nunique(dropna=True),
            'top_share': vc.iloc[0] if len(vc) > 0 else np.nan
        }
    rows.append(row)

summary = pd.DataFrame(rows)

def sig_round(x, n=4):
    if pd.isna(x):
        return x
    return float(f"{x:.{n}g}")

for col in ['mean', 'sd', 'min', 'median', 'max', 'top_share']:
    summary[col] = summary[col].apply(sig_round)

summary = summary.head(50)
summary = summary[['variable', 'type', 'group', 'mean', 'sd', 'min', 'median', 'max', 'n_missing', 'n_levels', 'top_share']]

mlca_runtime.save_table(summary, "distributions.csv", caption="Distribution summary of feature columns")

n_numeric = int((summary['type'] == 'numeric').sum())
n_categorical = int((summary['type'] == 'categorical').sum())
mlca_runtime.record_fact_candidate('n_numeric', n_numeric, source='distributions.csv')
mlca_runtime.record_fact_candidate('n_categorical', n_categorical, source='distributions.csv')

print(f"Features: {len(feature_cols)}, numeric: {n_numeric}, categorical: {n_categorical}")
print(f"Saved distributions.csv with {len(summary)} rows")
