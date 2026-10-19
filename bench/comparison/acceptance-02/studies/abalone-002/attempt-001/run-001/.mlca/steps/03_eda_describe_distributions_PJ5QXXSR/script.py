import mlca_runtime
import pandas as pd
import numpy as np
import json

mlca_runtime.set_seed(42)
inputs = mlca_runtime.inputs()

# Load protocol
protocol = json.loads(inputs['_protocol'].read_text())
targets = protocol['targets']
excluded = protocol.get('excluded_columns', [])
group = protocol.get('group_column', None)

# Load data
df = pd.read_parquet(inputs['abalone_uci'])

# Determine feature columns
all_cols = list(df.columns)
feat_cols = [c for c in all_cols if c not in targets and c not in excluded and c != group]

# Drop key-like columns: text codes with all distinct values
key_like = []
for c in feat_cols:
    if df[c].dtype == 'object' or df[c].dtype.name == 'string':
        if df[c].nunique(dropna=False) == len(df):
            key_like.append(c)
feat_cols = [c for c in feat_cols if c not in key_like]

# Placeholder handling for text columns
placeholders = {'?', '-', 'NA', 'N/A', ''}
ph_cols = []
for c in feat_cols:
    if df[c].dtype == 'object' or df[c].dtype.name == 'string':
        s = df[c].astype(str).str.strip()
        mask = s.isin(placeholders)
        if mask.any():
            ph_cols.append(c)
            df[c] = s.where(~mask, np.nan)
print(f"Placeholders found in {len(ph_cols)} columns")

# Classify columns
num_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(df[c])]
cat_cols = [c for c in feat_cols if c not in num_cols and (df[c].dtype == 'object' or df[c].dtype.name in ('string', 'category'))]

# Build summary
rows = []
for c in num_cols:
    s = df[c]
    rows.append({
        'column': c, 'type': 'numeric',
        'n_missing': int(s.isna().sum()),
        'mean': s.mean(), 'sd': s.std(ddof=1),
        'min': s.min(), 'median': s.median(), 'max': s.max(),
        'n_levels': np.nan, 'top_share': np.nan
    })
for c in cat_cols:
    s = df[c]
    vc = s.value_counts(dropna=True)
    n_lev = int(vc.shape[0])
    top_share = float(vc.iloc[0] / vc.sum()) if n_lev > 0 else np.nan
    rows.append({
        'column': c, 'type': 'categorical',
        'n_missing': int(s.isna().sum()),
        'mean': np.nan, 'sd': np.nan, 'min': np.nan, 'median': np.nan, 'max': np.nan,
        'n_levels': n_lev, 'top_share': top_share
    })

summary = pd.DataFrame(rows)

def round_sig(x, sig=4):
    if pd.isna(x) or not isinstance(x, (int, float, np.floating)):
        return x
    if x == 0:
        return 0
    return round(x, sig - int(np.floor(np.log10(abs(x)))) - 1)

for col in ['mean', 'sd', 'min', 'median', 'max', 'top_share']:
    summary[col] = summary[col].apply(round_sig)

summary['n_missing'] = summary['n_missing'].astype(int)
summary['n_levels'] = summary['n_levels'].apply(lambda x: int(x) if pd.notna(x) else np.nan)

mlca_runtime.save_table(summary, "distributions.csv", caption="Distribution summary of feature columns")

mlca_runtime.record_fact_candidate("n_numeric", len(num_cols), source="distributions.csv")
mlca_runtime.record_fact_candidate("n_categorical", len(cat_cols), source="distributions.csv")

print(f"Features: {len(feat_cols)} (numeric={len(num_cols)}, categorical={len(cat_cols)})")
print(f"Saved distributions.csv with {len(summary)} rows")
