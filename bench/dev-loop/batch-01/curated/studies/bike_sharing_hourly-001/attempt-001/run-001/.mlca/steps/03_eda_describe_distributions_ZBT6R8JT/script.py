import json
import numpy as np
import pandas as pd
import mlca_runtime

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()
df = pd.read_parquet(inputs['bike_sharing_hourly_uci'])
protocol = json.loads(inputs['_protocol'].read_text())

required = ['targets', 'excluded_columns', 'group_column', 'task']
missing = [k for k in required if k not in protocol]
if missing:
    raise ValueError(f"Missing keys {missing}. Available: {list(protocol.keys())}")

targets = protocol['targets']
excluded = protocol['excluded_columns']
group_col = protocol['group_column']
_ = protocol['task']

all_cols = list(df.columns)
feature_cols = [c for c in all_cols if c not in targets and c not in excluded and c != group_col]

def is_key_like(s):
    if pd.api.types.is_integer_dtype(s):
        return s.notna().all() and s.nunique() == len(s)
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        return s.notna().all() and s.nunique() == len(s)
    return False

feature_cols = [c for c in feature_cols if not is_key_like(df[c])]

rows = []
for c in feature_cols:
    s = df[c]
    n_missing = int(s.isna().sum())
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        rows.append({
            'variable': c, 'type': 'numeric', 'n_missing': n_missing,
            'mean': s.mean(), 'sd': s.std(), 'min': s.min(),
            'median': s.median(), 'max': s.max(),
            'n_levels': np.nan, 'top_share': np.nan
        })
    else:
        vc = s.value_counts(dropna=False)
        n_levels = int(s.nunique(dropna=False))
        top_share = vc.iloc[0] / len(s) if len(vc) > 0 else np.nan
        rows.append({
            'variable': c, 'type': 'categorical', 'n_missing': n_missing,
            'mean': np.nan, 'sd': np.nan, 'min': np.nan,
            'median': np.nan, 'max': np.nan,
            'n_levels': n_levels, 'top_share': top_share
        })

dist = pd.DataFrame(rows)
for col in ['n_missing', 'mean', 'sd', 'min', 'median', 'max', 'n_levels', 'top_share']:
    if col in dist.columns:
        dist[col] = dist[col].astype(float).round(4)

mlca_runtime.save_table(dist, 'distributions.csv', caption='Column distributions')

n_numeric = int((dist['type'] == 'numeric').sum())
n_categorical = int((dist['type'] == 'categorical').sum())
mlca_runtime.record_fact_candidate('n_numeric', n_numeric, source='distributions.csv')
mlca_runtime.record_fact_candidate('n_categorical', n_categorical, source='distributions.csv')

placeholders = ['?', '-', 'NA', 'N/A']
found = {}
for c in df.columns:
    if df[c].dtype == object or pd.api.types.is_string_dtype(df[c]):
        s = df[c].astype(str)
        for p in placeholders:
            if (s.str.strip() == p).any():
                found.setdefault(p, set()).add(c)
        if (s.str.strip() == '').any():
            found.setdefault('whitespace', set()).add(c)
if found:
    parts = [f"{k}: {len(v)} cols" for k, v in found.items()]
    print("Placeholders found: " + ", ".join(parts))
else:
    print("Placeholders found: none")

print(f"Features: {len(feature_cols)} (numeric={n_numeric}, categorical={n_categorical})")
print(f"Saved distributions.csv with {len(dist)} rows")
