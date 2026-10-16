import mlca_runtime
import pandas as pd
import numpy as np
import json

mlca_runtime.set_seed(42)

inputs = mlca_runtime.inputs()
protocol = json.loads(inputs['_protocol'].read_text())
required = ['targets', 'excluded_columns', 'group_column', 'task']
missing = [k for k in required if k not in protocol]
if missing:
    raise ValueError(f"Missing keys {missing} in _protocol. Available: {list(protocol.keys())}")

targets = protocol['targets']
excluded = protocol['excluded_columns']
group = protocol['group_column']

df = pd.read_parquet(inputs['seeds_uci'])

drop_cols = set(targets) | set(excluded)
if group is not None:
    drop_cols.add(group)

feature_cols = [c for c in df.columns if c not in drop_cols]

# Remove key-like columns: text codes or integer sequences whose values are all distinct and carry no measurement.
key_like = []
for c in feature_cols:
    if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]):
        if df[c].nunique(dropna=False) == len(df) and df[c].notna().all():
            key_like.append(c)
feature_cols = [c for c in feature_cols if c not in key_like]

# Check placeholders in text columns
placeholder_tokens = {'?', '-', 'NA', 'N/A', ''}
placeholder_cols = 0
found_tokens = set()
for c in feature_cols:
    if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]):
        stripped = df[c].astype(str).str.strip()
        for tok in placeholder_tokens:
            if (stripped == tok).any():
                found_tokens.add(tok)
                placeholder_cols += 1
                break
if placeholder_cols > 0:
    print(f"Placeholders found in {placeholder_cols} text columns: {sorted(found_tokens)}")

def round_sig(x, sig=4):
    if pd.isna(x):
        return x
    return float(f"{x:.{sig}g}")

rows = []
for c in feature_cols:
    if pd.api.types.is_numeric_dtype(df[c]):
        rows.append({
            'variable': c,
            'type': 'numeric',
            'mean': round_sig(df[c].mean()),
            'sd': round_sig(df[c].std()),
            'min': round_sig(df[c].min()),
            'median': round_sig(df[c].median()),
            'max': round_sig(df[c].max()),
            'n_missing': int(df[c].isna().sum()),
            'n_levels': np.nan,
            'top_level_share': np.nan
        })
    else:
        vc = df[c].value_counts(normalize=True, dropna=True)
        rows.append({
            'variable': c,
            'type': 'categorical',
            'mean': np.nan,
            'sd': np.nan,
            'min': np.nan,
            'median': np.nan,
            'max': np.nan,
            'n_missing': np.nan,
            'n_levels': int(df[c].nunique(dropna=True)),
            'top_level_share': round_sig(vc.iloc[0]) if len(vc) > 0 else np.nan
        })

summary = pd.DataFrame(rows)
mlca_runtime.save_table(summary, "distributions.csv", caption="Feature distributions summary")

n_numeric = int((summary['type'] == 'numeric').sum())
n_categorical = int((summary['type'] == 'categorical').sum())
mlca_runtime.record_fact_candidate('n_numeric', n_numeric, source='distributions.csv')
mlca_runtime.record_fact_candidate('n_categorical', n_categorical, source='distributions.csv')

print(f"Features: {n_numeric} numeric, {n_categorical} categorical")
print(f"Saved distributions.csv with {len(summary)} rows")
